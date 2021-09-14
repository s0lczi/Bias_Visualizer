import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import plotly.express as px
import re
import seaborn as sns
import warnings
from collections import Counter
from numpy import random
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Lipinski, rdMolDescriptors, Descriptors, rdMolHash, Descriptors3D, Fragments, QED
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem.rdchem import Mol
from sklearn.cluster import MeanShift, estimate_bandwidth
from scipy.stats import gaussian_kde
from typing import List, Dict, TextIO

warnings.filterwarnings("ignore")
matplotlib.use('Agg')


def prep_data(batch: str, out_path: str, actives_path: str, decoys_path: str) -> None:
    """ Prepares data to be analyzed by splitting files to right directories

    Args:
        batch (str): name of the analyzed dataset
        actives_path (str): path to the actives file
        decoys_path: path to the decoys file
        out_path: path were the output will be saved
    """
    if actives_path is not None:
        actives = Chem.SDMolSupplier(f'{actives_path}')
    else:
        actives = []
    if decoys_path is not None:
        decoys = Chem.SDMolSupplier(f'{decoys_path}')
    else:
        decoys = []

    split_file(actives, batch, 'actives', out_path)
    split_file(decoys, batch, 'decoys', out_path)


def split_file(molecules_set: list, batch: str, subset: str, out_path: str) -> None:
    """Splits large files with molecules into smaller ones basing on their SourceTag

    Args:
        molecules_set (list): list of molecules
        batch (str): name of the analyzed dataset
        subset (str): whether splitting actives or decoys
        out_path (str): path were the output will be saved
    """
    source_tag_in_processing = ""
    for molecule_number, single_molecule in enumerate(molecules_set):
        try:
            current_source_tag = single_molecule.GetProp("SourceTag")
            if current_source_tag != source_tag_in_processing:
                source_tag_in_processing = current_source_tag
                if not os.path.exists(f'{out_path}/{batch}/data/{source_tag_in_processing}'):
                    os.makedirs(f'{out_path}/{batch}/data/{source_tag_in_processing}')
                writer = Chem.SDWriter(f'{out_path}/{batch}/data/{source_tag_in_processing}/{subset}.sdf')
                writer.write(single_molecule)
            else:
                writer.write(single_molecule)
        except KeyError:
            fail_file = open(f"{out_path}/{batch}/data/no_target_{subset}.txt", 'a')
            fail_file.write(f'No SourceTag for molecule: {single_molecule.GetProp("_Name")} \n')
            fail_file.close()
        except AttributeError:
            fail_file = open(f"{out_path}/{batch}/data/bugs_{subset}.txt", 'a')
            fail_file.write(f'Bug in molecule number: {molecule_number} in target: {source_tag_in_processing} \n')
            fail_file.close()


def get_targets_list(batch: str, out_path: str) -> List[str]:
    """Gets list of targets that are about to being analyzed

    Args:
        batch (str): name of the analyzed dataset
        out_path (str): path were the output will be saved

    Returns:
        valid_targets (List): list of targets
    """
    valid_targets = []
    path_to_data = f'{out_path}/{batch}/data'
    targets_in_data = next(os.walk(path_to_data))[1]
    for single_target in targets_in_data:
        valid_targets.append(single_target)
    return valid_targets


def prep_input_summary(batch: str, out_path: str, targets: List[str]):
    """Prepares csv file with input summary (how many actives and decoys for each target)
    Args:
        batch:
        out_path:
        targets:
    """
    molecule_count = count_molecules(batch, out_path, targets)
    summary = pd.DataFrame(molecule_count).transpose()
    summary.to_csv(f'{out_path}/{batch}/data/input_summary.csv')


def concat_data(targets: list, batch: str, file: str, out_path: str) -> None:
    """Concatenates files associated with specific targets into one big file with all targets data

    Args:
        targets (str): list of targets being analyzed
        batch (str): name of the analyzed dataset
        file (str): name of file to concatenate
        out_path (str): path were the output will be saved

    """
    df_list = []
    for target in targets:
        df = pd.read_csv(f"{out_path}/{batch}/data/{target}/{file}", index_col=0, na_filter=False)
        df_list.append(df)
    result_df = pd.concat(df_list, keys=targets)
    result_df.to_csv(f"{out_path}/{batch}/data/all_{file}")


def make_matplotlib_colors(targets: List[str]) -> List:
    """Makes list of colors associated with specific targets

    Args:
        targets (List): list of targets being analyzed

    Returns:
        colors_list (List): list of colors for matplotlib figures
    """
    random.seed(156291)
    colors_list = [random.rand(3, ) for single_target in range(len(targets))]
    return colors_list


def make_plotly_colors(targets: List[str]) -> Dict:
    """Makes dictionary of colors associated with specific targets

    Args:
        targets (List): list of targets being analyzed

    Returns:
        plotly_dict (Dict): list of colors for plotly figures
    """
    plotly_dict = {}
    random.seed(156291)
    for single_target in targets:
        record = random.rand(3, )
        if single_target not in plotly_dict.keys():
            plotly_dict[single_target] = f'rgb({record[0]}, {record[1]}, {record[2]})'
    return plotly_dict


def make_seaborn_colors(targets: List[str]) -> Dict:
    """Makes dictionary of colors associated with specific targets

    Args:
        targets (List): list of targets being analyzed

    Returns:
        seaborn_dict (Dict): list of colors for seaborn figures
    """
    seaborn_dict = {}
    random.seed(156291)
    for single_target in targets:
        record = random.rand(3, )
        if single_target not in seaborn_dict.keys():
            seaborn_dict[single_target] = '#%02x%02x%02x' % (
                int(record[0] * 255), int(record[1] * 255), int(record[2] * 255))
    return seaborn_dict


def get_ring_systems(mol: Mol, includeSpiro: bool = False) -> List:
    """Counts ring systems in given molecule.

    Args:
        mol (Mol): molecules in which ring systems will be counted
        includeSpiro: whether to include spiro compunds

    Returns:
        systems (List): list of ring systems
    """
    ri = mol.GetRingInfo()
    systems = []
    for ring in ri.AtomRings():
        ring_ats = set(ring)
        n_systems = []
        for system in systems:
            n_in_common = len(ring_ats.intersection(system))
            if n_in_common and (includeSpiro or n_in_common > 1):
                ring_ats = ring_ats.union(system)
            else:
                n_systems.append(system)
        n_systems.append(ring_ats)
        systems = n_systems
    return systems


def count_elements(ligand) -> Dict:
    """Counts specific atoms in given molecule.

    Args:
        ligand (Mol): molecules in which atoms will be counted

    Returns:
        counter (Dict): dictionary with counted atoms
    """
    atoms = [atom.GetSymbol() for atom in ligand.GetAtoms()]
    counter = Counter(atoms)
    return counter


def feats_counter(targets: List[str], batch: str, subset: str, out_path: str) -> None:
    """Counts 15 different features and 11 different element occurances for every molecule in a given dataset.
    Then saves counted values in a .csv file.

    Args:
        targets (List): list of targets being analyzed
        batch (str): name of the analyzed dataset
        subset (str): whether splitting actives or decoys
        out_path (str): path were the output will be saved
    """
    organic_elements = ["B", "C", "N", "O", "P", "S", "F", "Cl", "Br", "I"]
    for target in targets:
        if not os.path.exists(f'{out_path}/{batch}/data/{target}/feats_2D_{subset}.csv'):
            data = Chem.SDMolSupplier(f'{out_path}/{batch}/data/{target}/{subset}.sdf')
            df = pd.DataFrame(columns=["HBA", "HBD", "ROT", "MW", "LogP", "NET", "ALL",
                                       "HEAVY", "CHI", "RING"])
            for ligand in range(len(data)):
                if data[ligand]:
                    atom_count = count_elements(data[ligand])
                    exotic = 0
                    for atom in data[ligand].GetAtoms():
                        if atom.GetSymbol() not in organic_elements:
                            exotic += 1
                    df = df.append(pd.Series({"HBA": Chem.Lipinski.NumHAcceptors(data[ligand]),
                                              "HBD": Chem.Lipinski.NumHDonors(data[ligand]),
                                              "ROT": Chem.Lipinski.NumRotatableBonds(data[ligand]),
                                              "MW": Chem.Descriptors.MolWt(data[ligand]),
                                              "LogP": Chem.Descriptors.MolLogP(data[ligand]),
                                              "NET": int(Chem.rdMolHash.MolHash(data[ligand],
                                                                                Chem.rdMolHash.HashFunction.NetCharge)),
                                              "ALL": data[ligand].GetNumAtoms(onlyExplicit=False),
                                              "HEAVY": data[ligand].GetNumHeavyAtoms(),
                                              "CHI": len(Chem.FindMolChiralCenters(data[ligand])),
                                              "RING": len(get_ring_systems(data[ligand])),
                                              "AmineT": Chem.Fragments.fr_NH0(data[ligand]),
                                              "AmineS": Chem.Fragments.fr_NH1(data[ligand]),
                                              "AmineP": Chem.Fragments.fr_NH2(data[ligand]),
                                              "CarbAcid": Chem.Fragments.fr_COO(data[ligand]),
                                              "HydrAcid": Chem.Fragments.fr_Ar_OH(
                                                  data[ligand]) + Chem.Fragments.fr_Al_OH(
                                                  data[ligand]),
                                              "B": atom_count["B"],
                                              "C": atom_count["C"],
                                              "N": atom_count["N"],
                                              "O": atom_count["O"],
                                              "P": atom_count["P"],
                                              "S": atom_count["S"],
                                              "F": atom_count["F"],
                                              "Cl": atom_count["Cl"],
                                              "Br": atom_count["Br"],
                                              "I": atom_count["I"],
                                              "Exotic": exotic},
                                             name=data[ligand].GetProp("_Name")))
                else:
                    pass
            df.to_csv(f"{out_path}/{batch}/data/{target}/feats_2D_{subset}.csv")
        else:
            pass


def feats_counter_3D(targets: List[str], batch: str, subset: str, out_path: str) -> None:
    """Counts 12 different 3D features for every molecule in a given dataset.
    Then saves counted values in a .csv file.

    Args:
        targets (List): list of targets being analyzed
        batch (str): name of the analyzed dataset
        subset (str): whether splitting actives or decoys
        out_path (str): path were the output will be saved
    """
    for target in targets:
        if not os.path.exists(f'{out_path}/{batch}/data/{target}/feats_3D_{subset}.csv'):
            df = pd.DataFrame(columns=["ASP", "ECC", "ISF", "NPR1", "NPR2", "PMI1", "PMI2",
                                       "PMI3", "ROG", "SI", "TPSA"])
            data = Chem.SDMolSupplier(f'{out_path}/{batch}/data/{target}/{subset}.sdf')

            for ligand in range(len(data)):
                if data[ligand]:
                    df = df.append(pd.Series({"ASP": round(Chem.Descriptors3D.Asphericity(data[ligand]), 2),
                                              "ECC": round(Chem.Descriptors3D.Eccentricity(data[ligand]), 2),
                                              "ISF": round(Chem.Descriptors3D.InertialShapeFactor(data[ligand]), 2),
                                              "NPR1": round(Chem.Descriptors3D.NPR1(data[ligand]), 2),
                                              "NPR2": round(Chem.Descriptors3D.NPR2(data[ligand]), 2),
                                              "PMI1": round(Chem.Descriptors3D.PMI1(data[ligand]), 2),
                                              "PMI2": round(Chem.Descriptors3D.PMI2(data[ligand]), 2),
                                              "PMI3": round(Chem.Descriptors3D.PMI3(data[ligand]), 2),
                                              "ROG": round(Chem.Descriptors3D.RadiusOfGyration(data[ligand]), 2),
                                              "SI": round(Chem.Descriptors3D.SpherocityIndex(data[ligand]), 2),
                                              "TPSA": round(Chem.rdMolDescriptors.CalcTPSA(data[ligand], force=False,
                                                                                           includeSandP=False), 2),
                                              "QED": round(Chem.QED.qed(data[ligand]), 2)},
                                             name=data[ligand].GetProp("_Name")))
                else:
                    pass
            df.to_csv(f"{out_path}/{batch}/data/{target}/feats_3D_{subset}.csv")
        else:
            pass


def prepare_data_for_clustering(data: pd.DataFrame, targets: List[str]) -> pd.DataFrame:
    """Counts mean, std and all three quartiles for every physicochemical feature for every target in a given DataFrame.
    Then saves them to a DataFrame.

    Args:
        data (DataFrame): DataFrame with data to count
        targets (List): list of targets being analyzed

    Returns:
        df (DataFrame): multiIndex DataFrame with counted values
    """
    result = []
    for target in targets:
        desc = data.loc[target].describe().transpose()
        result.append(desc)
    df = pd.concat(result, keys=targets)
    df = df.drop(["count", "min", "max"], 1)
    df.columns = ["MEAN", "STD", "Q1", "Q2", "Q3"]
    return df


def clustering(data: pd.DataFrame, targets_or_features: List[str], target_or_feature: str, measure_or_feature: str,
               q_val: float, lvl: int or None = None) -> Dict:
    """Clusters given data by chosen values. Either molecules by feature values or targets by overall feature values.

    Args:
        data (DataFrame): DataFrame with data to cluster
        targets_or_features (List): targets or features names
        target_or_feature (str): target or feature to perform the clustering on
        measure_or_feature (str): values that will be base for clustering
        q_val (float): value telling the algorithm how tight the clusters should be, the bigger the larger clusters
        lvl (int or None): whether clustering molecules (default: None) or targets by feature values (lvl = 1)

    Returns:
        result_dict (Dict): dictionary with clusters, cluster members,
         value range and count of cluster members in every cluster
    """
    result_dict = {}

    feature = data.xs(target_or_feature, level=lvl).round(2)

    values = np.array(
        list(zip(feature[measure_or_feature].tolist(), np.zeros(len(feature[measure_or_feature].tolist())))),
        dtype=np.float)
    targets_or_features = np.array(list(zip(targets_or_features, np.zeros(len(targets_or_features)))))

    bandwidth = estimate_bandwidth(values, quantile=q_val)
    while bandwidth == 0.0 and q_val != 0.5:
        q_val += 0.1
        bandwidth = estimate_bandwidth(values, quantile=q_val)
    ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
    ms.fit(values)
    labels = ms.labels_

    labels_unique = np.unique(labels)
    n_clusters_ = len(labels_unique)

    for k in range(n_clusters_):
        my_members = labels == k
        if f'Cluster {k + 1}' not in result_dict.keys():
            result_dict[f'Cluster {k + 1}'] = list(targets_or_features[my_members, 0])
            result_dict[f'Cluster {k + 1}'].append(
                f'{measure_or_feature} values in range from {np.amin(values[my_members, 0])}'
                f' to {np.amax(values[my_members, 0])}. Number of targets in this cluster: {len(values[my_members, 0])}')

    return result_dict


def filtrate(Act: pd.DataFrame, actives: pd.DataFrame, decoys: pd.DataFrame, target: str, feature: str, mol_number: int,
             positive: None or pd.Series) -> None or pd.Series:
    """Check if Series representing a specific molecule is valid or not

    Args:
        Act (DataFrame): data associated with all active molecules
        actives (DataFrame): data associated with active molecules of target being analyzed
        decoys (DataFrame): data associated with decoy molecules of target being analyzed
        target (str): name of the target being analyzed
        feature (str): name of the feature being analyzed
        mol_number (int): number of the molecule being analyzed
        positive (None or Series): valid molecules Series or None

    Returns:
        positive (None or Series): valid molecules Series or None
    """
    try:
        clusters = clustering(Act, list(range(len(actives))), target, feature, 0.1)
    except ValueError:
        positive = None
        return positive
    for cluster in clusters:
        if float(clusters[cluster][-1].split()[5]) <= decoys.loc[mol_number][feature] <= float(
                clusters[cluster][-1].split()[7]):
            positive = decoys.loc[mol_number]
            break
        else:
            positive = None
    if positive is None:
        return positive
    return positive


def perfect_filtering(Act: pd.DataFrame, Dec: pd.DataFrame, batch: str, out_path: str,
                      dimension: str, targets: List[str], feats: List[str]) -> pd.DataFrame:
    """Filtrates DataFrame to keep only perfectly matched decoys

    Args:
        Act (DataFrame): data associated with all active molecules
        Dec (DataFrame): data associated with all decoy molecules
        batch (str): name of the analyzed dataset
        out_path (str): path were the output will be saved
        dimension (str): whether data associated with 2D or 3D
        targets (List): list of targets being analyzed
        feats (List): list of features to consider in filtering

    Returns:
        result_df (DataFrame): filtered decoys dataset
    """
    filtered = []
    if not os.path.exists(f"{out_path}/{batch}/data/filtered_decoys_{dimension}.csv"):
        for target in targets:
            filtered = get_filtered_data(Act, Dec, feats, filtered, target)
        result_df = pd.concat(filtered, keys=targets)
        result_df.to_csv(f"{out_path}/{batch}/data/filtered_decoys_{dimension}.csv")
        return result_df
    else:
        result_df = pd.read_csv(f"{out_path}/{batch}/data/filtered_decoys_{dimension}.csv",
                                index_col=[0, 1], keep_default_na=False)
        return result_df


def get_filtered_data(Act: pd.DataFrame, Dec: pd.DataFrame, feats: List[str], filtered: List, target: str,
                      positive=None) -> List[pd.DataFrame]:
    """Gathers filtered DataFrame for given target

    Args:
        Act (DataFrame): data associated with all active molecules
        Dec (DataFrame): data associated with all decoy molecules
        feats (List): list of features to consider in filtering
        filtered (List): list of DataFrames of filtered targets
        target (str): target processing at the moment
        positive (None): filtering always starts as 'non valid'

    Returns:
        filtered (List): list of DataFrames of filtered targets
    """
    actives = Act.loc[target]
    decoys = Dec.loc[target]
    df = pd.DataFrame()
    for mol_number in range(len(decoys)):
        for feature in feats:
            positive = filtrate(Act, actives, decoys, target, feature, mol_number, positive)
            if positive is None:
                break
        if positive is not None:
            df = df.append(positive)
    filtered.append(df)
    return filtered


def calculate_data_for_plot(data_to_calculate: str, **kwargs) -> None:
    """Runs counting function basing on data_to_calculate value

    Args:
        data_to_calculate (str): which function to call
        **kwargs: kwargs associated to a specific function
    """
    if data_to_calculate == "feats_2D":
        feats_counter(kwargs["targets"], kwargs["batch"], kwargs["subset"], kwargs["out_path"])
    elif data_to_calculate == "feats_3D":
        feats_counter_3D(kwargs["targets"], kwargs["batch"], kwargs["subset"], kwargs["out_path"])
    elif data_to_calculate == "scaffolds_identical":
        count_scaffolds(kwargs["targets"], kwargs["batch"], kwargs["subset"], kwargs["out_path"])
    else:
        raise Exception('Unrecognized data type value.')


def concat_data_for_plot(data_to_plot: str, out_path: str, **kwargs) -> pd.DataFrame:
    """Calls concat_data function and returns DataFrame from  created .csv file

    Args:
        data_to_plot (str): which function to call
        out_path (str): path were the output will be saved
        **kwargs: kwargs associated to a specific function

    Returns:
        concatenated_dataframe (DataFrame): result from concat_data function
    """
    if data_to_plot.startswith("feats"):
        concat_data(kwargs["targets"], kwargs["batch"], f'{data_to_plot}_{kwargs["subset"]}.csv', out_path)
        concatenated_dataframe = pd.read_csv(f'{out_path}/{kwargs["batch"]}/'
                                             f'data/all_{data_to_plot}_{kwargs["subset"]}.csv',
                                             index_col=[0, 1], keep_default_na=False)
    elif data_to_plot == "scaffolds_identical":
        concat_data(kwargs["targets"], kwargs["batch"], f'count_scaffolds_{kwargs["subset"]}.csv', out_path)
        concatenated_dataframe = pd.read_csv(f'{out_path}/{kwargs["batch"]}/'
                                             f'data/all_count_scaffolds_{kwargs["subset"]}.csv',
                                             index_col=[0, 1], keep_default_na=False)
    elif data_to_plot.endswith("Tc"):
        concat_data(kwargs["targets"], kwargs["batch"],
                    f'{data_to_plot}_{kwargs["subset1"]}_vs_{kwargs["subset2"]}.csv', out_path)
        concatenated_dataframe = pd.read_csv(f'{out_path}/{kwargs["batch"]}/data/'
                                             f'all_{data_to_plot}_{kwargs["subset1"]}_vs_{kwargs["subset2"]}.csv',
                                             index_col=[0, 1], keep_default_na=False)
    else:
        raise Exception('Unrecognized data type value.')

    return concatenated_dataframe


def save_matplotlib_figs(plot_type: str, out_path: str, **kwargs) -> None:
    """Saves plots generated by matplotlib library

    Args:
        plot_type (str): where to save the plot
        out_path (str): path were the output will be saved
        **kwargs: kwargs associated to a specific function
    """
    if plot_type.startswith('Histograms'):
        if not os.path.exists(f"{out_path}/{kwargs['batch']}/plots/artificial_enrichment_bias/"
                              f"{plot_type}/{kwargs['dimension']}/{kwargs['subset']}"):
            os.makedirs(f"{out_path}/{kwargs['batch']}/plots/artificial_enrichment_bias/"
                        f"{plot_type}/{kwargs['dimension']}/{kwargs['subset']}")
        plt.savefig(f"{out_path}/{kwargs['batch']}/plots/artificial_enrichment_bias/"
                    f"{plot_type}/{kwargs['dimension']}/{kwargs['subset']}/{kwargs['base']}.pdf")
        plt.close()
    elif plot_type.startswith('KDE'):
        if not os.path.exists(f"{out_path}/{kwargs['batch']}/plots/artificial_enrichment_bias/"
                              f"{plot_type}/{kwargs['dimension']}"):
            os.makedirs(f"{out_path}/{kwargs['batch']}/plots/artificial_enrichment_bias/"
                        f"{plot_type}/{kwargs['dimension']}")
        plt.savefig(f"{out_path}/{kwargs['batch']}/plots/artificial_enrichment_bias/"
                    f"{plot_type}/{kwargs['dimension']}/{kwargs['base']}.pdf")
        plt.close()
    else:
        raise Exception("Unrecognized plot type. This won't be saved.")


def simple_feature_histograms_PF(targets: List[str], batch: str, subset: str, dimension: str,
                                 out_path: str) -> None:
    """Plots simple histograms of feature distribution within every target and saves them to a .pdf files.

    Args:
        targets (List): list of targets being analyzed
        batch (str): name of the analyzed dataset
        subset (str): whether splitting actives or decoys
        dimension (str): whether data associated with 2D or 3D
        out_path (str): path were the output will be saved
    """
    calculate_data_for_plot(f"feats_{dimension}", targets=targets, batch=batch, subset=subset, out_path=out_path)

    dataframe_to_plot = concat_data_for_plot(f"feats_{dimension}", out_path,
                                             targets=targets, batch=batch, subset=subset)

    colors_list = make_matplotlib_colors(targets)
    feats = list(dataframe_to_plot.columns)

    mins = []
    maxes = []
    for feat in feats:
        mins.append(min(dataframe_to_plot[feat]))
        maxes.append(max(dataframe_to_plot[feat]))

    for feat in range(len(feats)):
        fig, axs = plt.subplots(int(len(targets) / 3), 3, figsize=(16, 50), sharey='all', sharex='all',
                                tight_layout=True)
        axs = axs.ravel()
        xmin = mins[feat]
        xmax = maxes[feat]
        for target in range(len(targets)):
            axs[target].hist(dataframe_to_plot.loc[targets[target]][dataframe_to_plot.columns[feat]],
                             color=colors_list[target], range=(xmin, xmax))
            axs[target].set_title(f'Count of {feats[feat]} in target {targets[target]}', fontsize=12)
            axs[target].set_xlabel('Values', fontsize=8)
            axs[target].set_ylabel('Count', fontsize=8)
            axs[target].xaxis.set_tick_params(which='both', labelbottom=True)

        save_matplotlib_figs('Histograms_PF', out_path,
                             batch=batch, subset=subset, dimension=dimension, base=feats[feat])


def simple_feature_histograms_PT(targets: List[str], batch: str, subset: str, dimension: str,
                                 out_path: str) -> None:
    """Plots simple histograms of all features distributions for every target and saves them to a .pdf files.

    Args:
        targets (List): list of targets being analyzed
        batch (str): name of the analyzed dataset
        subset (str): whether splitting actives or decoys
        dimension (str): whether data associated with 2D or 3D
        out_path (str): path were the output will be saved
    """
    calculate_data_for_plot(f"feats_{dimension}", targets=targets, batch=batch, subset=subset, out_path=out_path)

    dataframe_to_plot = concat_data_for_plot(f"feats_{dimension}", out_path,
                                             targets=targets, batch=batch, subset=subset)

    colors_list = make_matplotlib_colors(targets)
    feats = list(dataframe_to_plot.columns)

    mins = []
    maxes = []
    for feat in feats:
        mins.append(min(dataframe_to_plot[feat]))
        maxes.append(max(dataframe_to_plot[feat]))

    for target in range(len(targets)):
        if dimension == "2D":
            fig, axs = plt.subplots(7, 4, figsize=(20, 20), sharey='all', tight_layout=True)
        elif dimension == "3D":
            fig, axs = plt.subplots(3, 4, figsize=(17, 10), sharey='all', tight_layout=True)
        else:
            raise Exception("Wrong dimension")
        axs = axs.ravel()
        for feat in range(len(feats)):
            xmin = mins[feat]
            xmax = maxes[feat]
            axs[feat].hist(dataframe_to_plot.loc[targets[target]][dataframe_to_plot.columns[feat]],
                           color=colors_list[target], range=(xmin, xmax))
            axs[feat].set_title(f'Count of {dataframe_to_plot.columns[feat]} in target {targets[target]}', fontsize=12)
            axs[feat].set_xlabel('Values', fontsize=8)
            axs[feat].set_ylabel('Count', fontsize=8)
            axs[feat].xaxis.set_tick_params(which='both', labelbottom=True)

        if dimension == "2D":
            fig.delaxes(axs[26])
            fig.delaxes(axs[27])

        save_matplotlib_figs('Histograms_PT', out_path,
                             batch=batch, subset=subset, dimension=dimension, base=targets[target])


def calculate_kde(**kwargs):
    """Calculates KDE of given data

    Args:
        **kwargs: kwargs associated to a given data

    Returns:

    """
    x0 = kwargs["actives"].loc[kwargs["target"]][kwargs["feat"]]
    x1 = kwargs["decoys"].loc[kwargs["target"]][kwargs["feat"]]

    try:
        x0_dominant_val = list(x0.value_counts().to_dict().keys())[0]
        x0_dominant_number = x0.value_counts()[x0_dominant_val] / len(x0)
    except KeyError:
        x0_dominant_number = 0
    try:
        x1_dominant_val = list(x1.value_counts().to_dict().keys())[0]
        x1_dominant_number = x1.value_counts()[x1_dominant_val] / len(x1)
    except KeyError:
        x1_dominant_number = 0

    if x0_dominant_number >= 0.9 and x1_dominant_number >= 0.9:
        x0 = x0.append(pd.Series([float(x0_dominant_val) + 1]), ignore_index=True)
        kde0 = gaussian_kde(x0, bw_method=0.337)
        kde1 = gaussian_kde(x0, bw_method=0.337)
    elif x0_dominant_number >= 0.9 and x1_dominant_number < 0.9:
        x0 = x0.append(pd.Series([x1.max()]), ignore_index=True)
        kde0 = gaussian_kde(x0, bw_method=0.337)
        kde1 = gaussian_kde(x1, bw_method=0.337)
    elif x0_dominant_number < 0.9 and x1_dominant_number >= 0.9:
        x1 = x1.append(pd.Series([x0.max()]), ignore_index=True)
        kde0 = gaussian_kde(x0, bw_method=0.337)
        kde1 = gaussian_kde(x1, bw_method=0.337)
    else:
        kde0 = gaussian_kde(x0, bw_method=0.337)
        kde1 = gaussian_kde(x1, bw_method=0.337)

    xmin = min(x0.min(), x1.min())
    xmax = max(x0.max(), x1.max())
    dx = 0.2 * (xmax - xmin)  # add a 20% margin, as the kde is wider than the data
    xmin -= dx
    xmax += dx

    x = np.linspace(xmin, xmax, 100)

    kde0_x = kde0(x)
    kde1_x = kde1(x)
    inters_x = np.minimum(kde0_x, kde1_x)
    area_inters_x = np.trapz(inters_x, x)
    return x, kde0_x, kde1_x, inters_x, area_inters_x


def kde_feature_plots_PF(targets: List[str], batch: str, dimension: str, out_path: str) -> None:
    """Plots KDE of feature distribution within every target and saves them to a .pdf files.

    Args:
        targets (List): list of targets being analyzed
        batch (str): name of the analyzed dataset
        dimension (str): whether data associated with 2D or 3D
        out_path (str): path were the output will be saved
    """
    calculate_data_for_plot(f"feats_{dimension}", targets=targets, batch=batch, subset='actives', out_path=out_path)
    calculate_data_for_plot(f"feats_{dimension}", targets=targets, batch=batch, subset='decoys', out_path=out_path)
    actives_to_plot = concat_data_for_plot(f"feats_{dimension}", out_path,
                                           targets=targets, batch=batch, subset='actives')
    decoys_to_plot = concat_data_for_plot(f"feats_{dimension}", out_path,
                                          targets=targets, batch=batch, subset='decoys')

    colors_list = make_matplotlib_colors(targets)
    feats = list(actives_to_plot.columns)

    for feat_number, feat in enumerate(feats):
        fig, axs = plt.subplots(int(len(targets) / 3), 3, figsize=(16, 50), sharey='all', sharex='all',
                                tight_layout=True)
        axs = axs.ravel()
        for target_number, target in enumerate(targets):
            x, kde0_x, kde1_x, inters_x, area_inters_x = calculate_kde(
                actives=actives_to_plot, decoys=decoys_to_plot, target=target, feat=feat)
            if area_inters_x is not None:
                axs[target_number].plot(x, kde0_x, color=colors_list[target_number], label='Actives')
                axs[target_number].fill_between(x, kde0_x, 0, color=colors_list[target_number], alpha=0.2)
                axs[target_number].plot(x, kde1_x, color='b', label='Decoys')
                axs[target_number].fill_between(x, kde1_x, 0, color='b', alpha=0.2)
                axs[target_number].plot(x, inters_x, color='r')
                axs[target_number].fill_between(x, inters_x, 0, facecolor='none', edgecolor='r', hatch='xx',
                                                label='Intersection')

                handles, labels = axs[target_number].get_legend_handles_labels()
                labels[2] += f': {area_inters_x * 100:.1f} %'
                axs[target_number].legend(handles, labels)
                axs[target_number].set_title(f"KDE of {feat} in target {target} and its decoys", fontsize=12)
                axs[target_number].set_xlabel('Values', fontsize=8)
                axs[target_number].set_ylabel('Density', fontsize=8)
                axs[target_number].xaxis.set_tick_params(which='both', labelbottom=True)
            else:
                sns.kdeplot(ax=axs[target_number], data=actives_to_plot.loc[target][feat],
                            color=colors_list[target_number], label='Actives')
                sns.kdeplot(ax=axs[target_number], data=decoys_to_plot.loc[target][feat],
                            color="#024ef4", label='Decoys')
                axs[target_number].set_title(f'KDE of {feat} in target {target} and its decoys', fontsize=12)
                axs[target_number].set_xlabel('Values', fontsize=10)
                axs[target_number].set_ylabel('Density', fontsize=10)
                axs[target_number].xaxis.set_tick_params(which='both', labelbottom=True)

        save_matplotlib_figs('KDE_plots_PF', out_path,
                             batch=batch, dimension=dimension, base=feat)


def kde_feature_plots_PT(targets: List[str], batch: str, dimension: str, out_path: str) -> None:
    """Plots KDE of all features distributions for every target and saves them to a .pdf files.

    Args:
        targets (List): list of targets being analyzed
        batch (str): name of the analyzed dataset
        dimension (str): whether data associated with 2D or 3D
        out_path (str): path were the output will be saved
    """
    calculate_data_for_plot(f"feats_{dimension}", targets=targets, batch=batch, subset='actives', out_path=out_path)
    calculate_data_for_plot(f"feats_{dimension}", targets=targets, batch=batch, subset='decoys', out_path=out_path)
    actives_to_plot = concat_data_for_plot(f"feats_{dimension}", out_path,
                                           targets=targets, batch=batch, subset='actives')
    decoys_to_plot = concat_data_for_plot(f"feats_{dimension}", out_path,
                                          targets=targets, batch=batch, subset='decoys')

    colors_list = make_matplotlib_colors(targets)
    feats = list(actives_to_plot.columns)

    for target_number, target in enumerate(targets):
        if dimension == "2D":
            fig, axs = plt.subplots(7, 4, figsize=(25, 25), tight_layout=True)
        elif dimension == "3D":
            fig, axs = plt.subplots(3, 4, figsize=(28, 14), tight_layout=True)
        else:
            raise Exception("Wrong dimension")
        axs = axs.ravel()
        for feat_number, feat in enumerate(feats):
            x, kde0_x, kde1_x, inters_x, area_inters_x = calculate_kde(
                actives=actives_to_plot, decoys=decoys_to_plot, target=target, feat=feat)

            axs[feat_number].plot(x, kde0_x, color=colors_list[target_number], label='Actives')
            axs[feat_number].fill_between(x, kde0_x, 0, color=colors_list[target_number], alpha=0.2)
            axs[feat_number].plot(x, kde1_x, color='b', label='Decoys')
            axs[feat_number].fill_between(x, kde1_x, 0, color='b', alpha=0.2)
            axs[feat_number].plot(x, inters_x, color='r')
            axs[feat_number].fill_between(x, inters_x, 0, facecolor='none', edgecolor='r', hatch='xx',
                                          label='Intersection')

            handles, labels = axs[feat_number].get_legend_handles_labels()
            labels[2] += f': {area_inters_x * 100:.1f} %'
            axs[feat_number].legend(handles, labels)
            axs[feat_number].set_title(f"KDE of {feat} in target {target} and its decoys", fontsize=12)
            axs[feat_number].set_xlabel('Values', fontsize=10)
            axs[feat_number].set_ylabel('Density', fontsize=10)
            axs[feat_number].xaxis.set_tick_params(which='both', labelbottom=True)

        if dimension == "2D":
            fig.delaxes(axs[26])
            fig.delaxes(axs[27])

        save_matplotlib_figs('KDE_plots_PT', out_path,
                             batch=batch, dimension=dimension, base=target)


def cluster_data(clustering_choice: str, dataframe_to_plot: pd.DataFrame,
                 targets: List[str], feat: str) -> (Dict, int, str):
    """Clusters data basing on one of the available measures ( MEAN, STD, Q1, Q2, Q3 )

    Args:
        clustering_choice (str): whether user want to cluster data or not
        dataframe_to_plot (DataFrame): data to cluster
        targets (List): list of targets being analyzed
        feat (str): which feature will be clustered

    Returns:
        order_dict (Dict): order of data to visualize
        cluster_num (int): number of cluster chosen to visualize
        measure (str): measure used to cluster the data
    """
    order_dict = {}
    if clustering_choice == "Y":
        data = prepare_data_for_clustering(dataframe_to_plot, targets)
        print("Type a measure to cluster by: MEAN, STD, Q1, Q2, Q3 ")
        measure = input()
        if measure in ["MEAN", "STD", "Q1", "Q2", "Q3"]:
            clusters = clustering(data, targets, feat, measure, 0.2, 1)
            print("Choose cluster number to visualize:")
            for key in clusters.keys():
                print(f"{key}: {clusters[key][-1]}")
            cluster_num = input()

            for target in targets:
                if target in clusters[f'Cluster {cluster_num}']:
                    if target not in order_dict.keys():
                        order_dict[target] = dataframe_to_plot.loc[target][feat].mean()
        else:
            raise Exception("Unrecognized measure.")

    elif clustering_choice == "N":
        cluster_num, measure = 0, ""
        for target in targets:
            if target not in order_dict.keys():
                order_dict[target] = dataframe_to_plot.loc[target][feat].mean()
    else:
        raise Exception("Unrecognized option.")
    return order_dict, cluster_num, measure


def save_seaborn_figs(plot_type: str, **kwargs):
    """Saves plots generated by seaborn library

    Args:
        plot_type (str): where to save the plot
        **kwargs: kwargs associated to a specific function
    """
    if plot_type.startswith("Box"):
        if not os.path.exists(f'{kwargs["out_path"]}/{kwargs["batch"]}/'
                              f'plots/artificial_enrichment_bias/Boxplots/{kwargs["dimension"]}/{kwargs["subset"]}'):
            os.makedirs(f'{kwargs["out_path"]}/{kwargs["batch"]}/'
                        f'plots/artificial_enrichment_bias/Boxplots/{kwargs["dimension"]}/{kwargs["subset"]}')
        if kwargs["clustering_choice"] == "Y":
            plt.savefig(f'{kwargs["out_path"]}/{kwargs["batch"]}/'
                        f'plots/artificial_enrichment_bias/Boxplots/{kwargs["dimension"]}/'
                        f'{kwargs["subset"]}/{kwargs["feat"]}_Cluster_{kwargs["cluster_num"]}_{kwargs["measure"]}.pdf')
            plt.close()
        else:
            plt.savefig(f'{kwargs["out_path"]}/{kwargs["batch"]}/'
                        f'plots/artificial_enrichment_bias/Boxplots/'
                        f'{kwargs["dimension"]}/{kwargs["subset"]}/{kwargs["feat"]}.pdf')
            plt.close()
    elif plot_type.startswith("Swarm"):
        if not os.path.exists(f'{kwargs["out_path"]}/{kwargs["batch"]}/plots/'
                              f'artificial_enrichment_bias/{kwargs["type"]}_swarmplots/'
                              f'{kwargs["dimension"]}/{kwargs["feat"]}'):
            os.makedirs(f'{kwargs["out_path"]}/{kwargs["batch"]}/plots/'
                        f'artificial_enrichment_bias/{kwargs["type"]}_swarmplots/{kwargs["dimension"]}/{kwargs["feat"]}')
        plt.savefig(f'{kwargs["out_path"]}/{kwargs["batch"]}/plots/'
                    f'artificial_enrichment_bias/{kwargs["type"]}_swarmplots/{kwargs["dimension"]}/'
                    f'{kwargs["feat"]}/{kwargs["feat"]}_{kwargs["target"]}.pdf')
        plt.close()
    elif plot_type.startswith("Tc"):
        if not os.path.exists(f'{kwargs["out_path"]}/{kwargs["batch"]}/plots/{kwargs["bias"]}/Boxplots'):
            os.makedirs(f'{kwargs["out_path"]}/{kwargs["batch"]}/plots/{kwargs["bias"]}/Boxplots')
        plt.savefig(
            f'{kwargs["out_path"]}/{kwargs["batch"]}/plots/{kwargs["bias"]}/'
            f'Boxplots/Tc_{kwargs["threshold_to_show"]}_{kwargs["subset1"]}_vs_{kwargs["subset2"]}.pdf')
        plt.close()
    else:
        raise Exception("Unrecognized plot type. This won't be saved.")


def boxplots(feat: str, targets: List[str], batch: str, subset: str, dimension: str, out_path: str) -> None:
    """Makes boxplots of given feature distribution for every target or only selected cluster
    if clustering data is included and saves the plot as .pdf file.
    Order of boxplots is descending based on a mean value.

    Args:
        feat (str): which feature will be clustered
        targets (List): list of targets being analyzed
        batch (str): name of the analyzed dataset
        subset (str): whether splitting actives or decoys
        dimension (str): whether data associated with 2D or 3D
        out_path (str): path were the output will be saved
    """
    calculate_data_for_plot(f"feats_{dimension}", targets=targets, batch=batch, subset=subset, out_path=out_path)

    dataframe_to_plot = concat_data_for_plot(f"feats_{dimension}", out_path,
                                             targets=targets, batch=batch, subset=subset)

    seaborn_dict = make_seaborn_colors(targets)

    print("Do you want to cluster your data? [Y/N]")
    clustering_choice = input().upper()
    order_dict, cluster_num, measure = cluster_data(clustering_choice, dataframe_to_plot, targets, feat)

    rev_df = dataframe_to_plot.reset_index(level=[0, 1]).drop("level_1", 1).rename(columns={"level_0": "Target"})

    order_dict = {k: v for k, v in sorted(order_dict.items(), key=lambda item: item[1], reverse=True)}
    seaborn_dict_order = {k: seaborn_dict[k] for k in order_dict.keys()}

    plt.figure(figsize=(40, 10))
    seaborn_palette = sns.set_palette(sns.color_palette(seaborn_dict_order.values()))
    ax = sns.boxplot(data=rev_df, y=feat, x="Target", palette=seaborn_palette, order=order_dict)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right', fontsize='x-large')
    if clustering_choice == "Y":
        ax.axes.set_title(f"Boxplots of {feat} for targets from Cluster {cluster_num} clustered by {measure}",
                          fontsize=40)
    else:
        ax.axes.set_title(f"Boxplots of {feat} for all targets", fontsize=40)
    ax.set_ylabel(f"{feat}", fontsize=30)
    ax.set_xlabel("Targets", fontsize=30)
    ax.tick_params(labelsize=20)
    for tick, color in zip(ax.get_xticklabels(), seaborn_dict_order.values()): tick.set_color(color)

    save_seaborn_figs("Boxplots", clustering_choice=clustering_choice, out_path=out_path, batch=batch,
                      subset=subset, dimension=dimension, feat=feat, cluster_num=cluster_num, measure=measure)


def single_swarmplot(targets: List[str], batch: str, out_path: str, dimension: str, feature: str) -> None:
    """Makes single swarmplot of given feature distribution for all targets
    in active and decoy datasets and saves the plot as .pdf file.

    Args:
        targets (List): list of targets being analyzed
        batch (str): name of the analyzed dataset
        out_path (str): path were the output will be saved
        dimension (str): whether data associated with 2D or 3D
        feature: feature to visualize
    """
    calculate_data_for_plot(f"feats_{dimension}", targets=targets, batch=batch, subset='actives', out_path=out_path)
    calculate_data_for_plot(f"feats_{dimension}", targets=targets, batch=batch, subset='decoys', out_path=out_path)

    actives = concat_data_for_plot(f"feats_{dimension}", out_path,
                                   targets=targets, batch=batch, subset='actives')
    decoys = concat_data_for_plot(f"feats_{dimension}", out_path,
                                  targets=targets, batch=batch, subset='decoys')

    actives["Activity"] = 1
    decoys["Activity"] = 0

    for target in targets:
        plt.figure(figsize=(15, 12))
        ax = sns.swarmplot(data=decoys.loc[target], y=feature, color="#024ef4")
        ax2 = sns.swarmplot(data=actives.loc[target], y=feature, color="#ef194a")
        ax.axes.set_title(f"Swarmplot of {feature} for target {target} and its decoys", fontsize=20)

        save_seaborn_figs("Swarmplot", out_path=out_path, batch=batch, dimension=dimension, feat=feature, type="Single",
                          target=target)


def double_swarmplots(targets: List[str], batch: str, out_path: str, dimension: str, feature: str) -> None:
    """Makes two swarmplots where one is the default one and the second one is filtered by perfect_filtering function.
    This is made for every target, for selected feature. Outputs are saved into .pdf files.

    Args:

        targets (List): list of targets being analyzed
        batch (str): name of the analyzed dataset
        out_path (str): path were the output will be saved
        dimension (str): whether data associated with 2D or 3D
        feature (str): feature to visualize

    Returns:

    """
    calculate_data_for_plot(f"feats_{dimension}", targets=targets, batch=batch, subset='actives', out_path=out_path)
    calculate_data_for_plot(f"feats_{dimension}", targets=targets, batch=batch, subset='decoys', out_path=out_path)

    actives = concat_data_for_plot(f"feats_{dimension}", out_path,
                                   targets=targets, batch=batch, subset='actives')
    decoys = concat_data_for_plot(f"feats_{dimension}", out_path,
                                  targets=targets, batch=batch, subset='decoys')

    filtering_2D = ["HBA", "HBD", "ROT", "MW", "LogP", "ALL", "HEAVY", "RING", "C", "N", "O"]
    filtering_3D = ["ASP", "ECC", "NPR1", "NPR2", "PMI1", "PMI2", "PMI3", "ROG", "SI", "TPSA", "QED"]

    if dimension == "2D":
        filtered_decoys = perfect_filtering(actives, decoys, batch, out_path, dimension, targets, filtering_2D)
    elif dimension == "3D":
        filtered_decoys = perfect_filtering(actives, decoys, batch, out_path, dimension, targets, filtering_3D)
    else:
        raise Exception("Wrong dimension.")

    actives["Activity"] = 1
    decoys["Activity"] = 0
    filtered_decoys["Activity"] = 0
    for target in range(len(targets)):
        try:
            fig, axs = plt.subplots(1, 2, figsize=(25, 15), sharey='all', sharex='all', tight_layout=True)
            sns.swarmplot(ax=axs[0], data=decoys.loc[targets[target]], y=feature, color="#024ef4")
            sns.swarmplot(ax=axs[0], data=actives.loc[targets[target]], y=feature, color="#ef194a")
            axs[0].set_title(f"Swarmplot of {feature} for target {targets[target]} and {batch} decoys",
                             fontsize=22)
            axs[0].tick_params(labelsize=20)
            axs[0].set_ylabel(f'{feature}', fontsize=20)

            sns.swarmplot(ax=axs[1], data=filtered_decoys.loc[targets[target]], y=feature, color="#024ef4")
            sns.swarmplot(ax=axs[1], data=actives.loc[targets[target]], y=feature, color="#ef194a")
            axs[1].set_title(f"Swarmplot of {feature} for target {targets[target]} and filtered {batch} decoys",
                             fontsize=22)
            axs[1].tick_params(labelsize=20)
            axs[1].set_ylabel(f'{feature}', fontsize=20)

            save_seaborn_figs("Swarmplot", out_path=out_path, batch=batch, dimension=dimension, feat=feature,
                              type="Multi",
                              target=targets[target])
        except KeyError:
            with open(f'{out_path}/{batch}/plots/artificial_enrichment_bias/Multi_Swarmplots/{dimension}/{feature}/'
                      f'{dimension}_filtration_failed.txt', 'a') as fail:
                fail.write(targets[target] + '\n')


def make_molecules_dict(molecules: List[Mol]) -> Dict:
    """Makes a dictionary where molecule name is a key, and molecule is a key value

    Args:
        molecules (List): molecules to place in a dict

    Returns:
        molecules_dict (Dict): dictionary with molecules
    """
    molecules_dict = {}
    for single_molecule in molecules:
        if single_molecule:
            try:
                if single_molecule.GetProp("_Name") not in molecules_dict.keys():
                    molecules_dict[single_molecule.GetProp("_Name")] = [single_molecule]
            except AttributeError:
                pass
    return molecules_dict


def make_scaffolds(molecules: Dict) -> Dict:
    """Makes a dictionary where molecule name is a key,
     and key value is a list of molecules made from scaffolds, and scaffolds themselves

    Args:
        molecules (Dict): molecules to obtain scaffolds from

    Returns:
        scaffolds (Dict): dictionary with scaffold and molecules from scaffolds
    """
    scaffolds = {}
    for molecule_name, single_molecule in molecules.items():
        if single_molecule[0]:
            single_scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=single_molecule[0])
            if single_scaffold and molecule_name not in scaffolds.keys():
                scaffolds[molecule_name] = [Chem.MolFromSmiles(single_scaffold), single_scaffold]
            else:
                scaffolds[molecule_name] = None
    return scaffolds


def make_fingerprints(molecules: Dict) -> Dict:
    """Makes a dictionary where molecule name is a key, and molecule fingerprint is a key value

    Args:
        molecules (Dict): molecules to obtain fingerprints from

    Returns:
        fingerprints (Dict): dictionary with fingerprints
    """
    fingerprints = {}
    for molecule_name, single_molecule in molecules.items():
        if single_molecule and single_molecule[0]:
            if molecule_name not in fingerprints.keys():
                fingerprints[molecule_name] = AllChem.GetMorganFingerprint(single_molecule[0], 2)
        else:
            fingerprints[molecule_name] = None
    return fingerprints


def scaffolds_similarity(targets: List[str], batch: str, subset1: str, subset2: str, threshold: float,
                         similarity: str, out_path: str) -> None:
    """Determines molecules similarity or dissimilarity
     basing on Tanimoto Coefficint value obtained from comparing fingerprints

    Args:
        targets (List): list of targets being analyzed
        batch (str): name of the analyzed dataset
        subset1 (str): whether splitting actives or decoys as subset 1
        subset2 (str): whether splitting actives or decoys as subset 2
        threshold (float): value from which to save data
        similarity (str): whether search for similarity or dissimilarity
        out_path (str): path were the output will be saved
    """
    for single_target in targets:
        if not os.path.exists(f"{out_path}/{batch}/data/{single_target}/scaffolds_Tc_{subset1}_vs_{subset2}.csv"):

            pd.DataFrame(columns=["Mol1", "Scaffold1", "Mol2", "Scaffold2", "Tc", "Target"]).to_csv(
                f"{out_path}/{batch}/data/{single_target}/scaffolds_Tc_{subset1}_vs_{subset2}.csv")

            actives = Chem.SDMolSupplier(f'{out_path}/{batch}/data/{single_target}/{subset1}.sdf')
            decoys = Chem.SDMolSupplier(f'{out_path}/{batch}/data/{single_target}/{subset2}.sdf')

            active_molecules = make_molecules_dict(actives)
            decoy_molecules = make_molecules_dict(decoys)

            active_scaffolds = make_scaffolds(active_molecules)
            decoy_scaffolds = make_scaffolds(decoy_molecules)

            active_fingerprints = make_fingerprints(active_scaffolds)
            decoy_fingerprints = make_fingerprints(decoy_scaffolds)

            for active_iterator, (active_name, active_molecule) in enumerate(active_scaffolds.items()):
                for decoy_iterator, (decoy_name, decoy_molecule) in enumerate(decoy_scaffolds.items()):
                    if similarity == "Y":
                        if decoy_iterator > active_iterator:
                            if active_fingerprints[active_name] and decoy_fingerprints[decoy_name]:
                                Tc = round(DataStructs.TanimotoSimilarity(active_fingerprints[active_name],
                                                                          decoy_fingerprints[decoy_name]), 2)
                                scaff_df = pd.DataFrame({"Mol1": active_name,
                                                         "Scaffold1": active_molecule[1],
                                                         "Mol2": decoy_name,
                                                         "Scaffold2": decoy_molecule[1],
                                                         "Tc": Tc,
                                                         "Target": single_target}, index=[0])
                                if Tc >= float(threshold):
                                    scaff_df.to_csv(
                                        f"{out_path}/{batch}/data/{single_target}/scaffolds_Tc_{subset1}_vs_{subset2}.csv",
                                        mode='a', header=False)
                    elif similarity == "N":
                        if active_fingerprints[active_name] and decoy_fingerprints[decoy_name]:
                            Tc = round(DataStructs.TanimotoSimilarity(active_fingerprints[active_name],
                                                                      decoy_fingerprints[decoy_name]), 2)
                            scaff_df = pd.DataFrame({"Mol1": active_name,
                                                     "Scaffold1": active_molecule[1],
                                                     "Mol2": decoy_name,
                                                     "Scaffold2": decoy_molecule[1],
                                                     "Tc": Tc,
                                                     "Target": single_target}, index=[0])
                            if Tc <= float(threshold):
                                scaff_df.to_csv(
                                    f"{out_path}/{batch}/data/{single_target}/scaffolds_Tc_{subset1}_vs_{subset2}.csv",
                                    mode='a', header=False)

        else:
            pass


def count_scaffolds(targets: List[str], batch: str, subset: str, out_path: str) -> None:
    """Count occurances of identical scaffolds in a given dataset

    Args:
        targets (List): list of targets being analyzed
        batch (str): name of the analyzed dataset
        subset (str): whether splitting actives or decoys
        out_path (str): path were the output will be saved
    """
    for target in targets:
        if not os.path.exists(f"{out_path}/{batch}/data/{target}/count_scaffolds_{subset}.csv"):
            molecules = Chem.SDMolSupplier(f'{out_path}/{batch}/data/{target}/{subset}.sdf')
            molecules = make_molecules_dict(molecules)
            scaffolds = make_scaffolds(molecules)

            count_df = pd.DataFrame(columns=["Scaffold"])
            for single_scaffold in scaffolds:
                if scaffolds[single_scaffold] and scaffolds[single_scaffold][0]:
                    count_df = count_df.append({"Scaffold": scaffolds[single_scaffold][1]},
                                               ignore_index=True)
            count_df = count_df["Scaffold"].value_counts().rename_axis('Scaffold').reset_index(name='Count')
            count_df["Target"] = target
            count_df = count_df.sort_values(by="Count", ascending=False)
            count_df.to_csv(f"{out_path}/{batch}/data/{target}/count_scaffolds_{subset}.csv")
        else:
            pass


def scaffolds_barplot_PT(targets: List[str], batch: str, subset: str, out_path: str, scaffolds_active: int,
                         scaffolds_decoy: int) -> None:
    """Makes barplots representing identical scaffolds for every target in given dataset

    Args:
        targets (List): list of targets being analyzed
        batch (str): name of the analyzed dataset
        subset (str): whether splitting actives or decoys
        out_path (str): path were the output will be saved
        scaffolds_active (int): the number of identical scaffolds in active sets from which they will be to visualized
        scaffolds_decoy (int): the number of identical scaffolds in decoy sets from which they will be to visualized
    """
    calculate_data_for_plot("scaffolds_identical", targets=targets, batch=batch, subset=subset, out_path=out_path)
    scaffolds_dataframe = concat_data_for_plot("scaffolds_identical", out_path,
                                               targets=targets, batch=batch, subset=subset)

    if subset == 'actives':
        if scaffolds_active is None:
            scaffolds_dataframe = scaffolds_dataframe.loc[scaffolds_dataframe["Count"] >= 2]
        else:
            scaffolds_dataframe = scaffolds_dataframe.loc[scaffolds_dataframe["Count"] >= int(scaffolds_active)]
    elif subset == 'decoys':
        if scaffolds_decoy is None:
            scaffolds_dataframe = scaffolds_dataframe.loc[scaffolds_dataframe["Count"] >= 10]
        else:
            scaffolds_dataframe = scaffolds_dataframe.loc[scaffolds_dataframe["Count"] >= int(scaffolds_decoy)]

    valid_targets = scaffolds_dataframe["Target"].drop_duplicates().tolist()
    colors_list = make_seaborn_colors(targets)

    fig, axs = plt.subplots(int(len(valid_targets)), 1, figsize=(13, 2 * len(targets)), tight_layout=True, sharex='all')
    axs = axs.ravel()
    for target_number, single_target in enumerate(valid_targets):
        scaffolds_to_plot = scaffolds_dataframe.loc[single_target]
        scaffolds_to_plot.sort_values('Count', inplace=True, ascending=False)
        if not scaffolds_to_plot.empty:
            axs[target_number].barh(scaffolds_to_plot["Scaffold"], scaffolds_to_plot["Count"],
                                    color=colors_list[single_target])
            axs[target_number].set_title(f"Count of identical scaffolds for target {single_target}",
                                         fontsize=12)
            axs[target_number].xaxis.set_tick_params(which='both', labelbottom=True)
            axs[target_number].invert_yaxis()
        else:
            axs[target_number].set_axis_off()

    if not os.path.exists(f"{out_path}/{batch}/plots/analogue_bias/Barplots/{subset}"):
        os.makedirs(f"{out_path}/{batch}/plots/analogue_bias/Barplots/{subset}")
    plt.savefig(f"{out_path}/{batch}/plots/analogue_bias/Barplots/{subset}/scaffolds_recurring_PT.pdf")
    plt.close()


def scaffolds_barplot_recurring(targets: List[str], batch: str, subset: str, out_path: str, scaffolds_active: int,
                                scaffolds_decoy: int) -> None:
    """Makes barplots representing identical scaffolds for every target in given dataset
    that occur in more than on target

    Args:
        targets (List): list of targets being analyzed
        batch (str): name of the analyzed dataset
        subset (str): whether splitting actives or decoys
        out_path (str): path were the output will be saved
        scaffolds_active (int): the number of identical scaffolds in active sets from which they will be to visualized
        scaffolds_decoy (int): the number of identical scaffolds in decoy sets from which they will be to visualized
    """
    calculate_data_for_plot("scaffolds_identical", targets=targets, batch=batch, subset=subset, out_path=out_path)
    scaffolds_dataframe = concat_data_for_plot("scaffolds_identical", out_path,
                                               targets=targets, batch=batch, subset=subset)

    if subset == 'actives':
        if scaffolds_active is None:
            scaffolds_dataframe = scaffolds_dataframe.loc[scaffolds_dataframe["Count"] >= 2]
        else:
            scaffolds_dataframe = scaffolds_dataframe.loc[scaffolds_dataframe["Count"] >= int(scaffolds_active)]
    elif subset == 'decoys':
        if scaffolds_decoy is None:
            scaffolds_dataframe = scaffolds_dataframe.loc[scaffolds_dataframe["Count"] >= 10]
        else:
            scaffolds_dataframe = scaffolds_dataframe.loc[scaffolds_dataframe["Count"] >= int(scaffolds_decoy)]

    scaffolds_dataframe = scaffolds_dataframe[scaffolds_dataframe.duplicated(subset="Scaffold", keep=False)]

    plotly_dict = make_plotly_colors(targets)

    fig = px.bar(scaffolds_dataframe, x="Count", y="Scaffold", color="Target", color_discrete_map=plotly_dict,
                 orientation='h')
    fig.update_layout(title_text='Count of scaffolds recurring in more then one target',
                      xaxis_title_text='Count',
                      yaxis_title_text='Scaffold',
                      xaxis=dict(
                          tickmode='array',
                          tickangle=0,
                          tickfont=dict(color='black', size=10)),
                      yaxis={'categoryorder': 'total ascending'},
                      legend={'traceorder': 'normal'})

    if not os.path.exists(f"{out_path}/{batch}/plots/analogue_bias/Barplots/{subset}"):
        os.makedirs(f"{out_path}/{batch}/plots/analogue_bias/Barplots/{subset}")
    fig.write_html(f"{out_path}/{batch}/plots/analogue_bias/Barplots/{subset}/scaffolds_recurring_overall.html")


def molecules_similarity(targets: List[str], batch: str, subset1: str, subset2: str, threshold: float,
                         similarity: str, out_path: str) -> None:
    """Determines molecules similarity or dissimilarity
     basing on Tanimoto Coefficient value obtained from comparing fingerprints

    Args:
        targets (List): list of targets being analyzed
        batch (str): name of the analyzed dataset
        subset1 (str): whether splitting actives or decoys as subset 1
        subset2 (str): whether splitting actives or decoys as subset 2
        threshold (float): value from which to save data
        similarity (str): whether search for similarity or dissimilarity
        out_path (str): path were the output will be saved
    """
    for single_target in targets:
        if not os.path.exists(f"{out_path}/{batch}/data/{single_target}/molecules_Tc_{subset1}_vs_{subset2}.csv"):

            pd.DataFrame(columns=["Mol1", "Mol2", "Tc", "Target"]).to_csv(
                f"{out_path}/{batch}/data/{single_target}/molecules_Tc_{subset1}_vs_{subset2}.csv")

            actives = Chem.SDMolSupplier(f'{out_path}/{batch}/data/{single_target}/{subset1}.sdf')
            decoys = Chem.SDMolSupplier(f'{out_path}/{batch}/data/{single_target}/{subset2}.sdf')

            active_molecules = make_molecules_dict(actives)
            decoy_molecules = make_molecules_dict(decoys)

            active_fingerprints = make_fingerprints(active_molecules)
            decoy_fingerprints = make_fingerprints(decoy_molecules)

            for active_iterator, (active_name, active_molecule) in enumerate(active_molecules.items()):
                for decoy_iterator, (decoy_name, decoy_molecule) in enumerate(decoy_molecules.items()):
                    if similarity == "Y":
                        if decoy_iterator > active_iterator:
                            if active_fingerprints[active_name] and decoy_fingerprints[decoy_name]:
                                Tc = round(DataStructs.TanimotoSimilarity(active_fingerprints[active_name],
                                                                          decoy_fingerprints[decoy_name]), 2)
                                df = pd.DataFrame({"Mol1": active_name,
                                                   "Mol2": decoy_name,
                                                   "Tc": Tc,
                                                   "Target": single_target}, index=[0])
                                if Tc >= float(threshold):
                                    df.to_csv(
                                        f"{out_path}/{batch}/data/{single_target}/molecules_Tc_{subset1}_vs_{subset2}.csv",
                                        mode='a', header=False)
                    elif similarity == "N":
                        if active_fingerprints[active_name] and decoy_fingerprints[decoy_name]:
                            Tc = round(DataStructs.TanimotoSimilarity(active_fingerprints[active_name],
                                                                      decoy_fingerprints[decoy_name]), 2)
                            df = pd.DataFrame({"Mol1": active_name,
                                               "Mol2": decoy_name,
                                               "Tc": Tc,
                                               "Target": single_target}, index=[0])
                            if Tc <= float(threshold):
                                df.to_csv(
                                    f"{out_path}/{batch}/data/{single_target}/molecules_Tc_{subset1}_vs_{subset2}.csv",
                                    mode='a', header=False)
        else:
            pass


def count_unique_molecules(targets: List[str], batch: str, subset1: str, subset2: str, threshold: float,
                           similarity: str, out_path: str, data_type: str):
    """Counts molecules not present in any comparison within a given threshold.
    Args:
        targets (List): list of targets being analyzed
        batch (str): name of the analyzed dataset
        subset1 (str): whether splitting actives or decoys as subset 1
        subset2 (str): whether splitting actives or decoys as subset 2
        threshold (float): value from which to save data
        similarity (str): whether search for similarity or dissimilarity
        out_path (str): path were the output will be saved
        data_type (str): whether data of scaffolds or whole molecules
    """
    unique_count = pd.DataFrame(columns=['Count', 'Target'])
    for single_target in targets:
        molecules = list(Chem.SDMolSupplier(f'{out_path}/{batch}/data/{single_target}/{subset1}.sdf'))
        molecules_to_check = []
        for mol_iterator, single_molecule in enumerate(molecules):
            if single_molecule is None:
                pass
            else:
                molecules_to_check.append(single_molecule.GetProp('_Name'))
        comparison_data = pd.read_csv(f'{out_path}/{batch}/data/{single_target}/'
                                      f'{data_type}_Tc_{subset1}_vs_{subset2}.csv',
                                      index_col=[0], keep_default_na=False)
        if similarity == 'Y':
            filtered_Tc = comparison_data.loc[comparison_data['Tc'] >= float(threshold)]
            molecule_names = list(dict.fromkeys(filtered_Tc['Mol1'].tolist() + filtered_Tc['Mol2'].tolist()))
        elif similarity == 'N':
            filtered_Tc = comparison_data.loc[comparison_data['Tc'] <= float(threshold)]
            molecule_names = filtered_Tc['Mol1'].drop_duplicates()

        count = len(set(molecules_to_check).difference(molecule_names))
        unique_count = unique_count.append({"Count": count,
                                            "Target": single_target},
                                           ignore_index=True)
    unique_count.to_csv(f'{out_path}/{batch}/data/{data_type}_unique_{subset1}_vs_{subset2}.csv')

    plot_unique(batch, data_type, out_path, subset1, subset2, targets, threshold, unique_count)


def plot_unique(batch: str, data_type: str, out_path: str, subset1: str, subset2: str, targets: List[str],
                threshold: float, unique_count: pd.DataFrame):
    """Plots molecules not present in any comparison within a given threshold.
    Args:
        batch (str): name of the analyzed dataset
        data_type (str): whether data of scaffolds or whole molecules
        out_path (str): path were the output will be saved
        subset1 (str): whether splitting actives or decoys as subset 1
        subset2 (str): whether splitting actives or decoys as subset 2
        targets (List): list of targets being analyzed
        threshold (float): value from which to save data
        unique_count (DataFrame): data about molecules considered as unique
    """
    colors_per_target = make_plotly_colors(targets)
    fig = px.histogram(unique_count, x="Target", y="Count", color="Target",
                       color_discrete_map=colors_per_target)
    fig.update_xaxes(categoryorder="total descending",
                     tickangle=45,
                     tickfont=dict(
                         size=7.5,
                         color='black'
                     ))
    fig.update_layout(title_text=f'Count of unique molecules within {threshold} Tc threshold',
                      xaxis_title_text='Target',
                      yaxis_title_text='Count',
                      autosize=False,
                      width=1000,
                      height=800,
                      margin=dict(l=50, r=50, b=50, t=50, pad=10),
                      legend={'traceorder': 'normal'})
    if data_type == 'scaffolds':
        save_plotly_fig("Unique", "Histograms", out_path, plot=fig, data_type=data_type, batch=batch,
                        threshold=threshold, subset1=subset1, subset2=subset2, bias='analogue_bias')
    elif data_type == 'molecules':
        save_plotly_fig("Unique", "Histograms", out_path, plot=fig, data_type=data_type, batch=batch,
                        threshold=threshold, subset1=subset1, subset2=subset2, bias='domain_bias')


def save_plotly_fig(plot_name: str, plot_type: str, out_path: str, **kwargs) -> None:
    """Saves plots generated by plotly library

    Args:
        plot_name (str): core part of the plot name
        plot_type (str): categorizes plots
        out_path (str): path were the output will be saved
        **kwargs: kwargs associated to a specific plot type
    """
    if not os.path.exists(f'{out_path}/{kwargs["batch"]}/plots/{kwargs["bias"]}/{plot_type}'):
        os.makedirs(f'{out_path}/{kwargs["batch"]}/plots/{kwargs["bias"]}/{plot_type}')
    kwargs["plot"].write_html(f'{out_path}/{kwargs["batch"]}/plots/{kwargs["bias"]}/{plot_type}/'
                              f'{plot_name}_{kwargs["threshold"]}_{kwargs["subset1"]}_vs_{kwargs["subset2"]}.html')


def plot_th(targets: List[str], similarity_to_show: pd.DataFrame, threshold_to_show: float):
    """Makes an interactive histogram showing how many pairs from all targets have a given threshold

    Args:
        targets (List): list of targets being analyzed
        similarity_to_show (DataFrame): filtered data that will be shown
        threshold_to_show (float): lower or upper value of threshold

    Returns:
        fig: plotted figure
    """
    colors_per_target = make_plotly_colors(targets)

    fig = px.histogram(similarity_to_show, x="Tc", color="Target", color_discrete_map=colors_per_target)
    fig.update_layout(
        title_text=f'Count of molecules pairs with Tc in threshold {threshold_to_show}',
        xaxis_title_text='Tc',
        yaxis_title_text='Count',
        barmode='stack',
        autosize=False,
        width=1000,
        height=800,
        yaxis=dict(
            tickmode='array'),
        margin=dict(l=50, r=50, b=50, t=50, pad=10),
        legend={'traceorder': 'normal'})
    return fig


def plot_pt(targets: List[str], similarity_to_show: pd.DataFrame, threshold_to_show: float):
    """Makes an interactive histogram showing how many pairs from a given target have a given threshold

    Args:
        targets (List): list of targets being analyzed
        similarity_to_show (DataFrame): filtered data that will be shown
        threshold_to_show (float): lower or upper value of threshold

    Returns:
        fig: plotted figure
    """
    colors_per_target = make_plotly_colors(targets)
    fig = px.histogram(similarity_to_show, x="Target", color="Target",
                       color_discrete_map=colors_per_target)
    fig.update_xaxes(categoryorder="total descending",
                     tickangle=45,
                     tickfont=dict(
                         size=7.5,
                         color='black'
                     ))
    fig.update_layout(title_text=f'Count of targets with Tc in threshold {threshold_to_show}',
                      xaxis_title_text='Target',
                      yaxis_title_text='Count',
                      autosize=False,
                      width=1000,
                      height=800,
                      margin=dict(l=50, r=50, b=50, t=50, pad=10),
                      legend={'traceorder': 'normal'})
    return fig


def plot_box(targets: List[str], similarity_to_show: pd.DataFrame, threshold_to_show: float):
    """Makes boxplots of given Tc distribution for every target and saves the plot as .pdf file.
    Order of boxplots is descending based on a mean value.

    Args:
        targets (List): list of targets being analyzed
        similarity_to_show (DataFrame): filtered data that will be shown
        threshold_to_show (float): lower or upper value of threshold

    Returns:
        fig: plotted figure
    """
    colors_per_target = make_seaborn_colors(targets)

    order_dict = {}
    for target in list(similarity_to_show["Target"].drop_duplicates()):
        if target not in order_dict.keys():
            order_dict[target] = similarity_to_show.loc[target]["Tc"].mean()

    order_dict = {k: v for k, v in sorted(order_dict.items(), key=lambda item: item[1], reverse=True)}
    colors_per_target_in_order = {k: colors_per_target[k] for k in order_dict.keys()}

    plt.figure(figsize=(40, 10))
    customPalette = sns.set_palette(sns.color_palette(colors_per_target_in_order.values()))
    ax = sns.boxplot(data=similarity_to_show, y="Tc", x="Target", palette=customPalette, order=order_dict)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right', fontsize='x-large')
    ax.axes.set_title(f"Boxplots for all targets with Tc in threshold {threshold_to_show}", fontsize=40)
    ax.set_ylabel("Tc", fontsize=30)
    ax.set_xlabel("Targets", fontsize=30)
    ax.tick_params(labelsize=20)
    for tick, color in zip(ax.get_xticklabels(), colors_per_target_in_order.values()): tick.set_color(color)
    return plt


def Tc_plots(targets: List[str], batch: str, threshold_to_show: float, data_type: str, out_path: str,
             plot_type: str, **kwargs) -> None:
    """Calls chosen plotting function on a given dataset

    Args:
        targets (List): list of targets being analyzed
        batch (str): name of the analyzed dataset
        threshold_to_show (float): lower or upper value of threshold
        data_type (str): whether data corresponds to molecules or scaffolds
        out_path (str): path were the output will be saved
        plot_type (str): categorizes plots
        **kwargs: kwargs associated to a specific plot type
    """
    if data_type == "molecules":
        molecules_similarity(targets, batch, kwargs["subset1"], kwargs["subset2"],
                             kwargs["threshold"], kwargs["similarity"], out_path)
        bias = 'domain_bias'
    elif data_type == "scaffolds":
        scaffolds_similarity(targets, batch, kwargs["subset1"], kwargs["subset2"],
                             kwargs["threshold"], kwargs["similarity"], out_path)
        bias = 'analogue_bias'

    data_for_plot = concat_data_for_plot(f"{data_type}_Tc", out_path, targets=targets, batch=batch,
                                         subset1=kwargs["subset1"], subset2=kwargs["subset2"])

    if kwargs["similarity"] == 'Y':
        similarity_to_show = data_for_plot.loc[data_for_plot["Tc"] >= float(threshold_to_show)]
    elif kwargs["similarity"] == 'N':
        similarity_to_show = data_for_plot.loc[data_for_plot["Tc"] <= float(threshold_to_show)]

    if plot_type == "HistogramTcTh":
        fig = plot_th(targets, similarity_to_show, threshold_to_show)
    elif plot_type == "HistogramTcPT":
        fig = plot_pt(targets, similarity_to_show, threshold_to_show)
    elif plot_type == "BoxplotsTc":
        fig = plot_box(targets, similarity_to_show, threshold_to_show)
    else:
        raise Exception(f'{plot_type} is not a valid option.')

    if plot_type == "HistogramTcTh" or plot_type == "HistogramTcPT":
        save_plotly_fig(f"{plot_type}", "Histograms", out_path, plot=fig, data_type=data_type, batch=batch,
                        threshold=threshold_to_show, subset1=kwargs["subset1"], subset2=kwargs["subset2"], bias=bias)
    elif plot_type == "BoxplotsTc":
        save_seaborn_figs("Tc_Boxplots", out_path=out_path, batch=batch, threshold_to_show=threshold_to_show,
                          subset1=kwargs["subset1"], subset2=kwargs["subset2"], bias=bias)
    else:
        raise Exception(f'{plot_type} is not a valid option.')


def make_summary(targets: List[str], out_path: str, batch: str, **kwargs):
    """Prepares summary of whole analysis in a form of a heatmap, based on the analyzes selected by the user.

    Args:
        targets (List): list of targets being analyzed
        batch (str): name of the analyzed dataset
        out_path (str): path were the output will be saved
        **kwargs: kwargs associated with analyzes selected by the user
    """
    for parameter in ['dimensions', 'subsets', 'similarity_types', 'comparison_types']:
        if kwargs[parameter] is None:
            kwargs[parameter] = []

    if not os.path.exists(f'{out_path}/{batch}/data/summary.csv'):
        summary = {}

        molecules_count = count_molecules(batch, out_path, targets)

        if kwargs['subsets'] == ['actives', 'decoys']:
            for single_dimension in kwargs['dimensions']:
                summary = summary_aeb(batch, single_dimension, out_path, summary, targets)

        if kwargs['comparison_types'] is not []:
            summary = summary_ab(batch, molecules_count, out_path, summary, targets, kwargs['comparison_types'])

        for similarity in kwargs['similarity_types']:
            if similarity == 'S':
                similarity = 'scaffolds'
            elif similarity == 'M':
                similarity = 'molecules'
            summary = summary_db(batch, molecules_count, out_path, similarity, summary, targets,
                                 kwargs['comparison_types'])

        summary_df = pd.concat(summary.values(), keys=summary.keys())
        summary_df.to_csv(f'{out_path}/{batch}/data/summary.csv')
    else:
        summary_df = pd.read_csv(f'{out_path}/{batch}/data/summary.csv', index_col=[0, 1], keep_default_na=False)

    summary_df = sort_bias(summary_df, targets)

    y_label = get_labels(summary_df)

    make_heatmap(batch, out_path, summary_df, y_label)


def count_molecules(batch: str, out_path: str, targets: List[str]) -> Dict:
    """Counts molecules in actives and decoys for all targets, then returns them as a dictionary.

    Args:
        batch (str): name of the analyzed dataset
        out_path (str): path were the output will be saved
        targets (List): list of targets being analyzed

    Returns:
        molecules_count (Dict): nested dictionary where main keys are targets, minor keys are 'actives' and 'decoys'
        and values are molecule counts
    """
    molecules_count = {}

    for single_target in targets:
        if single_target not in molecules_count.keys():
            try:
                actives = Chem.SDMolSupplier(f'{out_path}/{batch}/data/{single_target}/actives.sdf')
            except OSError:
                actives = []
            try:
                decoys = Chem.SDMolSupplier(f'{out_path}/{batch}/data/{single_target}/decoys.sdf')
            except OSError:
                decoys = []
            molecules_count[single_target] = {'actives': len(actives),
                                              'decoys': len(decoys)}

    return molecules_count


def summary_aeb(batch: str, dimension: str, out_path: str, summary: Dict, targets: List[str]) -> Dict:
    """Determines the artificial enrichment bias level for the selected dimension, then saves the results to the
    DataFrame and finally places the obtained DataFrame in the dictionary with the key corresponding to the checked
    dimension

    Args:
        batch (str): name of the analyzed dataset
        dimension (str): currently analyzed dimension
        out_path (str): path were the output will be saved
        summary (Dict): whole analysis summary
        targets (List): list of targets being analyzed

    Returns:
        summary (Dict): whole analysis summary with current analyzed part added
    """
    actives = concat_data_for_plot(f"feats_{dimension}", out_path,
                                   targets=targets, batch=batch, subset='actives')
    decoys = concat_data_for_plot(f"feats_{dimension}", out_path,
                                  targets=targets, batch=batch, subset='decoys')
    if dimension == "2D":
        features = ["HBA", "HBD", "ROT", "MW", "LogP", "NET", "ALL", "HEAVY", "CHI", "RING",
                    "B", "C", "N", "O", "P", "S", "F", "Cl", "Br", "I", "Exotic",
                    "AmineT", "AmineS", "AmineP", "CarbAcid", "HydrAcid"]
    elif dimension == "3D":
        features = ["ASP", "ECC", "ISF", "NPR1", "NPR2", "PMI1", "PMI2", "PMI3", "ROG", "SI", "TPSA", "QED"]

    all_targets_aeb_summary = {}

    for single_target in targets:
        target_aeb_summary = {}
        for single_feature in features:
            kde_result = calculate_kde(actives=actives, decoys=decoys, target=single_target, feat=single_feature)
            coverage = round(kde_result[4] * 100)
            if coverage <= 50:
                target_aeb_summary[single_feature] = 1
            elif int(coverage) in range(51, 75):
                target_aeb_summary[single_feature] = 0.5
            else:
                target_aeb_summary[single_feature] = 0

        all_targets_aeb_summary[single_target] = target_aeb_summary
    summary[f'feats_{dimension}'] = pd.DataFrame(all_targets_aeb_summary)

    return summary


def summary_ab(batch: str, molecules_count: Dict, out_path: str, summary: Dict, targets: List[str],
               comparison_types: List[str]) -> Dict:
    """Determines the domain bias and partially analogue bias level for the selected comparison types, then saves the
    results to the DataFrame and finally places the obtained DataFrame in the dictionary with the key corresponding to
    the checked comparison
    Args:
        batch (str): name of the analyzed dataset
        molecules_count (Dict): dict with molecule counts for all targets
        out_path (str): path were the output will be saved
        summary (Dict): whole analysis summary
        targets (List): list of all targets being analyzed
        comparison_types: whether comparing actives vs actives, actives vs decoys, decoys vs decoys or all of them

    Returns:
        summary (Dict): whole analysis summary with current analyzed part added
    """
    all_targets_ab_summary = {}

    for single_target in targets:
        target_ab_summary = {}
        for comparison in comparison_types:
            if comparison == 'ava':
                sub = 'actives'
            elif comparison == 'dvd':
                sub = 'decoys'
            elif comparison == 'avd':
                continue
            target_data = pd.read_csv(f'{out_path}/{batch}/data/{single_target}/'
                                      f'scaffolds_Tc_{sub}_vs_{sub}.csv',
                                      index_col=[0, 1], keep_default_na=False)

            Tc_to_analyze = target_data.loc[target_data["Tc"] == 1]
            count = molecules_count[single_target][sub]
            biased_Tc_percent = round((len(Tc_to_analyze["Tc"]) / (((count * count) - count) / 2)) * 100)

            if biased_Tc_percent >= 20:
                target_ab_summary[comparison] = 1
            elif biased_Tc_percent in range(10, 19):
                target_ab_summary[comparison] = 0.5
            else:
                target_ab_summary[comparison] = 0

        all_targets_ab_summary[single_target] = target_ab_summary
    summary[f'scaffolds_identical'] = pd.DataFrame(all_targets_ab_summary)

    return summary


def summary_db(batch: str, molecules_count: Dict, out_path: str, similarity: str, summary: Dict, targets: List[str],
               comparison_types: List[str]) -> Dict:
    """Determines the domain bias and partially analogue bias level for the selected comparison types, then saves the
    results to the DataFrame and finally places the obtained DataFrame in the dictionary with the key corresponding to
    the checked comparison
    Args:
        batch (str): name of the analyzed dataset
        molecules_count (Dict): dict with molecule counts for all targets
        out_path (str): path were the output will be saved
        similarity: whether comparing scaffolds similarity, whole molecules similarity or both
        summary (Dict): whole analysis summary
        targets (List): list of all targets being analyzed
        comparison_types: whether comparing actives vs actives, actives vs decoys, decoys vs decoys or all of them

    Returns:
        summary (Dict): whole analysis summary with current analyzed part added
    """
    all_targets_db_summary = {}

    for single_target in targets:
        target_db_summary = {}
        for comparison in comparison_types:
            if comparison == "avd":
                sub1 = 'actives'
                sub2 = 'decoys'
            elif comparison == 'ava':
                sub1 = 'actives'
                sub2 = 'actives'
            elif comparison == 'dvd':
                sub1 = 'decoys'
                sub2 = 'decoys'
            target_data = pd.read_csv(f'{out_path}/{batch}/data/{single_target}/'
                                      f'{similarity}_Tc_{sub1}_vs_{sub2}.csv',
                                      index_col=[0, 1], keep_default_na=False)
            if comparison != 'avd':
                Tc_to_analyze = target_data.loc[target_data["Tc"] >= 0.8]
                count = molecules_count[single_target][sub1]
                biased_Tc_percent = round((len(Tc_to_analyze["Tc"]) / (((count * count) - count) / 2)) * 100)
            else:
                Tc_to_analyze = target_data.loc[target_data["Tc"] <= 0.4]
                actives_count = molecules_count[single_target][sub1]
                decoys_count = molecules_count[single_target][sub2]
                biased_Tc_percent = round((((actives_count * decoys_count) - len(Tc_to_analyze["Tc"])) / (
                        actives_count * decoys_count)) * 100)

            if similarity == "molecules":
                if biased_Tc_percent >= 30:
                    target_db_summary[comparison] = 1
                elif biased_Tc_percent in range(10, 29):
                    target_db_summary[comparison] = 0.5
                else:
                    target_db_summary[comparison] = 0
            elif similarity == "scaffolds":
                if biased_Tc_percent >= 40:
                    target_db_summary[comparison] = 1
                elif biased_Tc_percent in range(15, 39):
                    target_db_summary[comparison] = 0.5
                else:
                    target_db_summary[comparison] = 0

        all_targets_db_summary[single_target] = target_db_summary
    summary[f'{similarity}_Tc'] = pd.DataFrame(all_targets_db_summary)

    return summary


def sort_bias(summary_df: pd.DataFrame, targets: List[str]) -> pd.DataFrame:
    """Sorts the DataFrame with biased data in descending order of the number of biased parts

    Args:
        summary_df (DataFrame): whole analysis summary
        targets (List): list of all targets being analyzed

    Returns:
        summary_df (DataFrame): sorted analysis summary DataFrame
    """
    order_dict = {}
    for target in targets:
        try:
            high_bias = summary_df[target].value_counts()[1] * 3
        except TypeError:
            high_bias = 0
        except KeyError:
            high_bias = 0
        try:
            moderate_bias = summary_df[target].value_counts()[0.5]
        except TypeError:
            moderate_bias = 0
        except KeyError:
            moderate_bias = 0
        order_dict[target] = high_bias + moderate_bias
    order_dict = dict(sorted(order_dict.items(), key=lambda item: item[1], reverse=True))
    summary_df = summary_df[order_dict.keys()]
    return summary_df


def get_labels(summary_df: pd.DataFrame) -> List[str]:
    """Creates appropriate labels for the heatmap rows based on the MultiIndex DataFrame

    Args:
        summary_df (DataFrame): whole analysis summary

    Returns:
        x_label (List): labels for summary heatmap rows
    """
    all_labels = {'feats_2D--HBA': 'Features 2D: HBA (AEB)',
                  'feats_2D--HBD': 'HBD (AEB)',
                  'feats_2D--ROT': 'ROT (AEB)',
                  'feats_2D--MW': 'MW (AEB)',
                  'feats_2D--LogP': 'LogP (AEB)',
                  'feats_2D--NET': 'NET (AEB)',
                  'feats_2D--ALL': 'ALL (AEB)',
                  'feats_2D--HEAVY': 'HEAVY (AEB)',
                  'feats_2D--CHI': 'CHI (AEB)',
                  'feats_2D--RING': 'RING (AEB)',
                  'feats_2D--B': 'B (AEB)',
                  'feats_2D--C': 'C (AEB)',
                  'feats_2D--N': 'N (AEB)',
                  'feats_2D--O': 'O (AEB)',
                  'feats_2D--P': 'P (AEB)',
                  'feats_2D--S': 'S (AEB)',
                  'feats_2D--F': 'F (AEB)',
                  'feats_2D--Cl': 'Cl (AEB)',
                  'feats_2D--Br': 'Br (AEB)',
                  'feats_2D--I': 'I (AEB)',
                  'feats_2D--Exotic': 'Exotic (AEB)',
                  'feats_2D--AmineT': 'AmineT (AEB)',
                  'feats_2D--AmineS': 'AmineS (AEB)',
                  'feats_2D--AmineP': 'AmineP (AEB)',
                  'feats_2D--CarbAcid': 'CarbAcid (AEB)',
                  'feats_2D--HydrAcid': 'HydrAcid (AEB)',
                  'feats_3D--ASP': 'Features 3D: ASP (AEB)',
                  'feats_3D--ECC': 'ECC (AEB)',
                  'feats_3D--ISF': 'ISF (AEB)',
                  'feats_3D--NPR1': 'NPR1 (AEB)',
                  'feats_3D--NPR2': 'NPR2 (AEB)',
                  'feats_3D--PMI1': 'PMI1 (AEB)',
                  'feats_3D--PMI2': 'PMI2 (AEB)',
                  'feats_3D--PMI3': 'PMI3 (AEB)',
                  'feats_3D--ROG': 'ROG (AEB)',
                  'feats_3D--SI': 'SI (AEB)',
                  'feats_3D--TPSA': 'TPSA (AEB)',
                  'feats_3D--QED': 'QED (AEB)',
                  'scaffolds_identical--ava': 'Identical scaffolds in actives (AB)',
                  'scaffolds_identical--dvd': 'Identical scaffolds in decoys (AB)',
                  'scaffolds_Tc--ava': 'scaffolds Tc: actives vs actives (AB)',
                  'scaffolds_Tc--avd': 'scaffolds Tc: actives vs decoys (AB)',
                  'scaffolds_Tc--dvd': 'scaffolds Tc: decoys vs decoys (AB)',
                  'molecules_Tc--ava': 'Tc: actives vs actives (DB)',
                  'molecules_Tc--avd': 'Tc: actives vs decoys (DB)',
                  'molecules_Tc--dvd': 'Tc: decoys vs decoys (DB)',
                  }
    y_label = []
    for single_tuple in summary_df.index.values.tolist():
        y_label.append(all_labels['--'.join(single_tuple)])
    return y_label


def make_heatmap(batch, out_path, summary_df, y_label):
    """Produces a summary heatmap based on the received DataFrame

    Args:
        batch (str): name of the analyzed dataset
        out_path (str): path were the output will be saved
        summary_df (DataFrame): whole analysis summary
        y_label (List): labels for summary heatmap rows
    """
    fig = px.imshow(summary_df, x=summary_df.columns, y=y_label,
                    title=f'Heatmap of biased data in {batch}',
                    labels=dict(x="Targets", y="Analyzed Data", color="Bias"),
                    color_continuous_scale=['lightgrey', 'red'],
                    zmin=0, zmax=1
                    )
    fig.update_xaxes(tickangle=45,
                     tickfont=dict(
                         size=8,
                         color='black'
                     ))
    fig.update_yaxes(tickangle=0,
                     tickfont=dict(
                         size=9,
                         color='black'
                     ))
    fig.update_traces(xgap=0.8, ygap=0.8)
    fig.update_coloraxes(colorbar=dict(tickmode="array",
                                       nticks=4,
                                       tickvals=[0, 0.5, 1],
                                       ticktext=["LOW", "MODERATE", "HIGH"],
                                       ))
    fig.write_html(f'{out_path}/{batch}/plots/summary.html')


# # DEKOIS 2.0 - AD


def validation(targets: List[str], batch: str, out_path: str) -> [pd.DataFrame, List[str]]:
    """Checks if all molecules in a given dataset are valid.
    Valid means that molecules has at least one from three required values (IC50, Ki, Kd)

    Args:
        targets (List): list of targets being analyzed
        batch (str): name of the analyzed dataset
        out_path (str): path were the output will be saved

    Returns:
        valid (DataFrame): contains valid molecules
        invalid (List): contains invalid molecules name
    """
    valid = []
    invalid = []
    pattern = re.compile(r'^[0-9]*$')
    for target in targets:
        df = pd.DataFrame(columns=["ID", "MEASURE", "VALUE"])
        molecules = Chem.SDMolSupplier(f'{out_path}/{batch}/data/{target}/actives.sdf')
        for single_molecule in molecules:
            properties = single_molecule.GetPropsAsDict()
            if "Ki_in_nM" in properties.keys() and pattern.search(str(properties["Ki_in_nM"])):
                val = float(properties["Ki_in_nM"])
                meas = 'Ki'
            elif "Kd_in_nM" in properties.keys() and pattern.search(str(properties["Kd_in_nM"])):
                val = float(properties["Kd_in_nM"])
                meas = 'Kd'
            elif "IC50_in_nM" in properties.keys() and pattern.search(str(properties["IC50_in_nM"])):
                val = float(properties["IC50_in_nM"])
                meas = 'IC50'
            else:
                invalid.append(properties['Name'])
                meas = "None"
                val = 0
            df = df.append({"ID": properties['Name'],
                            "MEASURE": meas,
                            "VALUE": val},
                           ignore_index=True)
        valid.append(df)
    result = pd.concat(valid, keys=targets)
    result["ACTIVITY"] = 1
    return result, invalid


def split_sdf(targets: List[str], batch: str, out_path: str) -> None:
    """Splits large files with molecules into smaller ones basing on their SourceTag.
    Each molecules is placed in and individual file.

    Args:
        targets (List): list of targets being analyzed
        batch (str): name of the analyzed dataset
        out_path (str): path were the output will be saved
    """
    for target in targets:
        if not os.path.exists(f'{out_path}/{batch}/AD/{target}/molecules'):
            os.makedirs(f'{out_path}/{batch}/AD/{target}/molecules')
        molecules = Chem.SDMolSupplier(f'{out_path}/{batch}/data/{target}/actives.sdf')
        for single_molecule in molecules:
            writer = Chem.SDWriter(f'{out_path}/{batch}/AD/{target}/molecules/{single_molecule.GetProp("_Name")}.sdf')
            writer.write(single_molecule)
            writer.close()


def change_sourcetag(decoy: TextIO, targets: List[str], target: str, file_AD: TextIO) -> None:
    """Changes molecule SourceTag according to the new dataset it belongs to then saves output

    Args:
        decoy (TextIO): read decoy file
        targets (List): list of targets being analyzed
        target (str): target in processing
        file_AD (TextIO): AD output file
    """
    for line in decoy.readlines():
        try:
            if line.split()[0] in targets:
                line = target
        except IndexError:
            file_AD.write(line)
        file_AD.write(line)


def make_AD(data: pd.DataFrame, targets: List[str], batch: str, out_path: str) -> None:
    """Makes AD datasets basing on new SourceTags

    Args:
        data (DataFrame): data about valid molecules
        targets (List): list of targets being analyzed
        batch (str): name of the analyzed dataset
        out_path (str): path were the output will be saved
    """
    pd.set_option('mode.chained_assignment', None)
    df_list = []
    for target in targets:
        decoys = [x for x in targets if x != target]
        decoys_pd = data.loc[decoys]
        decoys_pd['ACTIVITY'] = 0
        AD = pd.concat([data.loc[target], decoys_pd]).drop_duplicates(subset='ID', keep='first')
        df_list.append(AD)
        decoys_AD = AD.loc[AD["ACTIVITY"] == 0]
        with open(f'{out_path}/{batch}/AD/{target}/decoys.sdf', 'w') as file_AD:
            for row in range(len(decoys_AD)):
                decoy_target = decoys_AD.index[row][0]
                decoy_name = decoys_AD["ID"][row]
                with open(f'{out_path}/{batch}/AD/{decoy_target}/molecules/{decoy_name}.sdf', 'r') as decoy:
                    change_sourcetag(decoy, targets, target, file_AD)
