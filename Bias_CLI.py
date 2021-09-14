import Bias_Visualizer as bv
import argparse
import textwrap as _textwrap
import re
from typing import List

SUBSETS = ["actives", "decoys"]
PHYSICOCHEMICAL_FEATURES_2D = ["HBA", "HBD", "ROT", "MW", "LogP", "NET", "ALL", "HEAVY", "CHI", "RING",
                               "B", "C", "N", "O", "P", "S", "F", "Cl", "Br", "I", "Exotic",
                               "AmineT", "AmineS", "AmineP", "CarbAcid", "HydrAcid"]
PHYSICOCHEMICAL_FEATURES_3D = ["ASP", "ECC", "ISF", "NPR1", "NPR2", "PMI1", "PMI2", "PMI3", "ROG", "SI", "TPSA", "QED"]
DIMENSION = ["2D", "3D"]
AEB_PLOTS = ["1", "2", "3", "4", "5", "6", "7"]
AB_PLOTS = ["8", "9"]
DB_PLOTS = ["10", "11", "12", "13"]
SIMILARITY_TYPES = ["M", "S"]
COMPARISON_TYPES = ["ava", "avd", "dvd"]
ALL_ANALYZES = {"1": "HistogramsPF",
                "2": "HistogramsPT",
                "3": "KDEplotsPF",
                "4": "KDEplotsPT",
                "5": "Boxplots",
                "6": "SingleSwarmplot",
                "7": "DoubleSwarmplot",
                "8": "BarplotsPT",
                "9": "BarplotsRecurring",
                "10": "HistogramTcPT",
                "11": "HistogramTcTh",
                "12": "HistogramUnique",
                "13": "BoxplotsTc"
                }


class PreserveWhiteSpaceWrapRawTextHelpFormatter(argparse.RawDescriptionHelpFormatter):
    def __add_whitespace(self, idx, iWSpace, text):
        if idx == 0:
            return text
        return (" " * iWSpace) + text

    def _split_lines(self, text, width):
        textRows = text.splitlines()
        for idx, line in enumerate(textRows):
            search = re.search('\s*[0-9\-]{0,}\.?\s*', line)
            if line.strip() == "":
                textRows[idx] = " "
            elif search:
                lWSpace = search.end()
                lines = [self.__add_whitespace(i, lWSpace, x) for i, x in enumerate(_textwrap.wrap(line, width))]
                textRows[idx] = lines

        return [item for sublist in textRows for item in sublist]


ap = argparse.ArgumentParser(formatter_class=PreserveWhiteSpaceWrapRawTextHelpFormatter)
ap.add_argument("-n", "--name", required=True,
                help="This is the name for the master directory, where all data for a given analysis is stored. This "
                     "allows to append new data to the existing analysis or conduct additional analysis for an "
                     "existing project.")
ap.add_argument("-o", "--output", required=True,
                help="The path to where we want our results to be saved.")
ap.add_argument("-in_a", "--actives", required=False,
                help="Path to the file containing the active molecules.")
ap.add_argument("-in_d", "--decoys", required=False,
                help="Path to the file containing the decoy molecules.")
ap.add_argument("-plt", "--plot_type", nargs='+', choices=AEB_PLOTS + AB_PLOTS + DB_PLOTS, required=False,
                metavar='PLOT_TYPE', help="The type of analysis to perform, which defines also the visualizations to "
                                          "produce. The user can choose from:\n1 – Histograms per feature, "
                                          "showing the distribution of a given feature in all targets.\n2 – "
                                          "Histograms per target, showing the distribution of all features in a given "
                                          "target.\n3 – KDE plots per feature, comparing KDE (with data intersection "
                                          "percent) of a given feature in both active and decoys sets for all "
                                          "targets.\n4 – KDE plots per target, comparing KDE (with data intersection "
                                          "percent) of all features in both active and decoys sets for a given "
                                          "target.\n5 – Boxplots, showing the distribution of selected feature(s) in "
                                          "all targets, with optional possibility to cluster data.\n6 – Single "
                                          "Swarmplot, showing the distribution of selected feature(s) in all targets "
                                          "for both active and decoy sets.\n7 – Double Swarmplot, showing the "
                                          "distribution of selected feature(s) in all targets for both active and "
                                          "decoy sets with additional filtering function to compare filtered and "
                                          "unfiltered decoy sets.\n8 – Barplot per target, representing identical "
                                          "scaffolds recurring in each target in active or decoy set.\n9 – Barplot "
                                          "collective for all targets with scaffolds occurring in more than one "
                                          "target in active or decoy set.\n10 – Histogram of Tc per target, "
                                          "showing how many pairs of molecules in each target have similarity in a "
                                          "given range.\n11 – Histogram of Tc per threshold, showing how many pairs "
                                          "of molecules for each target have their similarity equal to a specific "
                                          "value.\n12 – Histogram per target, showing how many molecules have "
                                          "similarity beyond a given range and therefore are unique.\n13 – Boxplots "
                                          "of Tc, showing distribution of Tc for all targets in a given threshold "
                                          "range.")
ap.add_argument("-dim", "--dimension", nargs='+', choices=DIMENSION, required=False,
                metavar='DIMENSION', help="Dimension is an argument related to the analysis of artificial enrichment "
                                          "bias. It is required for all figures related to this type of bias. It "
                                          "determines which physicochemical features we want to analyze 2D, "
                                          "3D or both.")
ap.add_argument("-feat", "--feature", nargs='+', choices=PHYSICOCHEMICAL_FEATURES_2D + PHYSICOCHEMICAL_FEATURES_3D,
                metavar='FEATURE', required=False, help="Feature is another argument related to AEB, in addition, "
                                                        "it is only used for figures 5-7. One can choose from any of "
                                                        "the 2D and 3D features listed in the paragraph about "
                                                        "physicochemical features.")
ap.add_argument("-sub", "--subset", nargs='+', choices=SUBSETS, required=False,
                metavar='SUBSET', help="Subset is an argument related to specific types of plots that are not "
                                       "produced collectively for active and decoy molecules. These are: 1, 2, 5, "
                                       "8 and 9. Choices here are actives, decoys, or both.")
ap.add_argument("-sa", "--scaffolds_active", required=False, help="Determines the number of identical scaffolds in "
                                                                  "active sets from which they will be to visualized "
                                                                  "in figures 8 and 9. The default value is 2.")
ap.add_argument("-sd", "--scaffolds_decoy", required=False, help="Determines the number of identical scaffolds in "
                                                                 "decoy sets from which they will be to visualized "
                                                                 "in figures 8 and 9. The default value is 10.")
ap.add_argument("-sim", "--similarity_type", nargs='+', choices=SIMILARITY_TYPES, required=False,
                metavar='SIMILARITY_TYPE', help="Similarity type must be specified to produce figures 10-13. Choices "
                                                "are: S for scaffolds, M for molecules, or both depending on whether "
                                                "we want to compare the scaffolds only or whole molecules.")
ap.add_argument("-vs", "--versus", nargs='+', choices=COMPARISON_TYPES, required=False,
                metavar='VERSUS', help="Versus determines what comparisons will be made for the figures 10-13. User "
                                       "can choose from: ava, avd and dvd which stands for actives VS actives, "
                                       "actives VS decoys and  decoys VS decoys.")
ap.add_argument("-tc_s", "--tanimoto_similarity", required=False,
                help="This value determines the Tc threshold for molecular similarity comparisons (ava, dvd). The "
                     "data write threshold is fixed at 0.5 Tc, which means that the user can specify the threshold "
                     "from 0.5 upwards. For low biased datasets, fixed threshold prevents the creation of outputs "
                     "that consume huge amounts of disk space.")
ap.add_argument("-tc_ds", "--tanimoto_dissimilarity", required=False,
                help="Similar to the value above determines the Tc threshold that will be plotted on the figures "
                     "comparing the different datasets (avd). The data write threshold is again fixed at 0.5, "
                     "in this case it means that the user can specify the threshold from 0.5 lower. For highly biased "
                     "datasets, a fixed threshold prevents the creation of outputs that consume huge amounts of disk "
                     "space.")
ap.add_argument("-all", "--all_in", required=False, action='store_true',
                help="All available types of analysis to be performed and all figures will be produced, except 5, 6, "
                     "7 which are not suitable for all types of data and require user attention.")
ap.add_argument("-ad", "--make_ad", required=False, action='store_true',
                help="Entering this argument will create an AD dataset from the active molecules involved in the "
                     "given analysis.")


def plot_aeb(targets: List, plot: str, name: str, output: str, dimensions: List, subsets: List, features: List,
             feats_2D: List, feats_3D: List):
    if plot == "HistogramsPF":
        for sub in subsets:
            for dim in dimensions:
                print(f"Calculating and plotting {dim} features HistogramsPF for {sub}.")
                bv.simple_feature_histograms_PF(targets, name, sub, dim, output)

    if plot == "HistogramsPT":
        for sub in subsets:
            for dim in dimensions:
                print(f"Calculating and plotting {dim} features HistogramsPT for {sub}.")
                bv.simple_feature_histograms_PT(targets, name, sub, dim, output)

    if plot == "KDEplotsPF":
        for dim in dimensions:
            print(f"Calculating and plotting {dim} features KDEplotsPF.")
            bv.kde_feature_plots_PF(targets, name, dim, output)

    if plot == "KDEplotsPT":
        for dim in dimensions:
            print(f"Calculating and plotting {dim} features KDEplotsPT.")
            bv.kde_feature_plots_PT(targets, name, dim, output)

    if plot == "Boxplots":
        for sub in subsets:
            for dim in dimensions:
                if dim == "2D":
                    for feat in features:
                        if feat in feats_2D:
                            print(f"Calculating and plotting {dim} features Boxplot for {sub}.")
                            bv.boxplots(feat, targets, name, sub, dim, output)
                if dim == "3D":
                    for feat in features:
                        if feat in feats_3D:
                            print(f"Calculating and plotting {dim} features Boxplot for {sub}.")
                            bv.boxplots(feat, targets, name, sub, dim, output)

    if plot == "SingleSwarmplot":
        for dim in dimensions:
            if dim == "2D":
                for feat in features:
                    if feat in feats_2D:
                        print(f"Calculating and plotting {dim} features Single Swarmplots.")
                        bv.single_swarmplot(targets, name, output, dim, feat)
            if dim == "3D":
                for feat in features:
                    if feat in feats_3D:
                        print(f"Calculating and plotting {dim} features Single Swarmplots.")
                        bv.single_swarmplot(targets, name, output, dim, feat)

    if plot == "DoubleSwarmplot":
        for dim in dimensions:
            if dim == "2D":
                for feat in features:
                    if feat in feats_2D:
                        print(f"Calculating and plotting {dim} features Double Swarmplots.")
                        bv.double_swarmplots(targets, name, output, dim, feat)
            if dim == "3D":
                for feat in features:
                    if feat in feats_3D:
                        print(f"Calculating and plotting {dim} features Double Swarmplots.")
                        bv.double_swarmplots(targets, name, output, dim, feat)


def plot_ab(plot, name, output, subsets, scaff_active, scaff_decoy):
    if plot == "BarplotsPT":
        for sub in subsets:
            print(f"Calculating and plotting BarplotsPT for {sub}.")
            bv.scaffolds_barplot_PT(targets, name, sub, output, scaff_active, scaff_decoy)

    if plot == "BarplotsRecurring":
        for sub in subsets:
            print(f"Calculating and plotting BarplotsRecurring for {sub}.")
            bv.scaffolds_barplot_recurring(targets, name, sub, output, scaff_active, scaff_decoy)


def plot_db(plot, name, output, comparisons, similarity_types, tanimoto_s, tanimoto_ds):
    for sim in similarity_types:
        if sim == 'S':
            sim = 'scaffolds'
        elif sim == 'M':
            sim = 'molecules'
        for comp in comparisons:
            if comp == "avd":
                sub1 = 'actives'
                sub2 = 'decoys'
                print(f"Calculating and plotting {plot} for {sim} {comp}.")
                if plot != 'HistogramUnique':
                    bv.Tc_plots(targets, batch=name, threshold_to_show=tanimoto_ds,
                                data_type=sim, out_path=output, plot_type=plot,
                                subset1=sub1, subset2=sub2, threshold=0.5, similarity="N")
                else:
                    bv.count_unique_molecules(targets, name, sub1, sub2, tanimoto_ds, 'N', output, sim)
            else:
                if comp == 'ava':
                    sub = 'actives'
                elif comp == 'dvd':
                    sub = 'decoys'
                print(f"Calculating and plotting {plot} for {sim} {comp}.")
                if plot != 'HistogramUnique':
                    bv.Tc_plots(targets, batch=name, threshold_to_show=tanimoto_s,
                                data_type=sim, out_path=output, plot_type=plot,
                                subset1=sub, subset2=sub, threshold=0.5, similarity="Y")
                else:
                    bv.count_unique_molecules(targets, name, sub, sub, tanimoto_s, 'Y', output, sim)


def assess_targets(name, output):
    targets = bv.get_targets_list(name, output)
    bv.prep_input_summary(name, output, targets)
    print('Files checked. Input summary ready.')
    return targets


args = ap.parse_args()

if not bv.os.path.exists(f'{args.output}/{args.name}') and (args.actives is None and args.decoys is None):
    print("Directory for this analysis doesn't exist. Please pass at least one of the the input files and try again.")

elif not bv.os.path.exists(f'{args.output}/{args.name}') and (args.actives is not None or args.decoys is not None):
    print("Preparing and checking files...")
    bv.prep_data(args.name, args.output, args.actives, args.decoys)
    targets = assess_targets(args.name, args.output)

elif bv.os.path.exists(f'{args.output}/{args.name}') and (args.actives is not None or args.decoys is not None):
    print("Preparing and checking files...")
    bv.prep_data(args.name, args.output, args.actives, args.decoys)
    targets = assess_targets(args.name, args.output)

elif bv.os.path.exists(f'{args.output}/{args.name}') and (args.actives is None or args.decoys is None):
    targets = assess_targets(args.name, args.output)

if args.all_in is True:
    args.subset = SUBSETS
    args.dimension = DIMENSION
    args.similarity_type = SIMILARITY_TYPES
    args.versus = COMPARISON_TYPES
    args.plot_type = ["1", "2", "3", "4"] + AB_PLOTS + DB_PLOTS
    args.tanimoto_similarity = 0.8
    args.tanimoto_dissimilarity = 0.4

if args.plot_type:
    for plot in args.plot_type:
        plot_name = ALL_ANALYZES[plot]
        if plot in AEB_PLOTS:
            plot_aeb(targets, plot_name, args.name, args.output, args.dimension, args.subset, args.feature,
                     PHYSICOCHEMICAL_FEATURES_2D, PHYSICOCHEMICAL_FEATURES_3D)
        if plot in AB_PLOTS:
            plot_ab(plot_name, args.name, args.output, args.subset, args.scaffolds_active, args.scaffolds_decoy)
        if plot in DB_PLOTS:
            similarity_type = sorted(args.similarity_type, reverse=True)
            plot_db(plot_name, args.name, args.output, args.versus, similarity_type,
                    args.tanimoto_similarity, args.tanimoto_dissimilarity)
    print('Preparing summary...')
    bv.make_summary(targets, args.output, args.name, dimensions=args.dimension, subsets=args.subset,
                    similarity_types=args.similarity_type, comparison_types=args.versus)

if args.make_ad is True:
    valid, invalid = bv.validation(targets, args.name, args.output)
    bv.split_sdf(targets, args.name, args.output)
    bv.make_AD(valid, targets, args.name, args.output)
