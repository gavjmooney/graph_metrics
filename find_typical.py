from metrics_suite import MetricsSuite
from matplotlib import pyplot as plt
import os
import time
import pandas as pd
import random
import numpy as np
from scipy.stats import linregress
import networkx as nx
from write_graph import write_graphml_pos

def typical():
    # result_df = pd.read_csv("GDM_with_diff.csv")
    result_df = pd.read_csv("GDM_with_diff.csv")
    result_df = result_df.set_index("filename")

    # result_df = result_df[result_df["n"] >= 70]
    # result_df = result_df[result_df["generator"] != "Rome"]
    # result_df = result_df[result_df["generator"] != "North"]
    result_df = result_df[result_df["layout"] == "HOLA"]
    # result_df = result_df[result_df["layout"] != "Kamada-Kawai"]

    #print(result_df)


    # print(result_df["num_within_range"].max())
    # print(result_df["num_within_range"].mean())
    # print(result_df["num_within_range"].median())
    # print(result_df["num_within_range"].max())
    # print()

    # print(result_df["abs_diff_sum"].min())
    # print(result_df["abs_diff_sum"].mean())
    # print(result_df["abs_diff_sum"].median())
    # print(result_df["abs_diff_sum"].max())

    labels = ['angular_resolution', 'aspect_ratio', 'crossing_angle', 'edge_crossings', 'edge_length',
        'edge_orthogonality', 'gabriel_ratio', 'neighbourhood_preservation', 'node_resolution',
        'node_uniformity']

    # #print(result_df.loc["HOLA_grafo1184.25.gml"]["abs_diff_sum"])
    # medians = []
    # for l in labels:
    #     medians.append(result_df.loc["HOLA_grafo1184.25.gml"][l].mean())
    # print(np.mean(medians))
    # print()
    # #print(result_df.loc["kamada-kawai_g.52.1.graphml"]["abs_diff_sum"])
    # medians = []
    # for l in labels:
    #     medians.append(result_df.loc["kamada-kawai_g.52.1.graphml"][l].mean())
    # print(np.mean(medians))
    # print()
    # #print(result_df.loc["fruchterman-reingold_j0_NWS_i21_n100_m144_p0.500.graphml"]["abs_diff_sum"])
    # medians = []
    # for l in labels:
    #     medians.append(result_df.loc["fruchterman-reingold_j0_NWS_i21_n100_m144_p0.500_k3.graphml"][l].mean())
    # print(np.mean(medians))
    # print()
    # quit()

    # medians = []
    # for l in labels:
    #     medians.append(result_df[l].median())

    # # print(medians)
    # print(np.mean(medians))
    # print()
    # quit()

    # result_df = result_df[result_df["generator"] != "Rome"]
    # result_df = result_df[result_df["generator"] != "North"]

    #max_num_within_range = result_df['num_within_range'].max()


    #max_rows = result_df[result_df['num_within_range'] == 6]



    lowest_abs_diff_sum_rows = result_df.nsmallest(50, 'abs_diff_sum')
    print(lowest_abs_diff_sum_rows)

    # # Extract the filenames with the lowest abs_diff_sum
    # filenames_lowest_abs_diff_sum = lowest_abs_diff_sum_rows['filename'].values

    # # Print the filenames with the lowest abs_diff_sum
    # # for filename in filenames_lowest_abs_diff_sum:
    # #     print(result_df[filename])

    # lowest_abs_diff_sum_rows = result_df.nsmallest(10, 'abs_diff_sum')

    # # Print the rows with the lowest abs_diff_sum
    # print(lowest_abs_diff_sum_rows)


    # Filter the dataframe to include only the rows with the maximum num_within_range
    # max_rows = result_df[result_df['num_within_range'] == max_num_within_range]
    # print(len(max_rows))

    # Sort the max_rows dataframe by ascending order of abs_diff_sum and get the 10 lowest rows
    #lowest_abs_diff_sum_rows = max_rows.nsmallest(50, 'abs_diff_sum')

    #lowest_abs_diff_sum_rows = result_df.nsmallest(60, 'abs_diff_sum')

    #lowest_abs_diff_sum_rows = result_df.nsmallest(50, 'best_diff_sum')


    # Print the rows with the lowest abs_diff_sum
    #print(lowest_abs_diff_sum_rows)



def numeric_distribution(filename):
    df = pd.read_csv(filename)
    #df = df.set_index("filename")

    # Create dataframes for each layout
    fr_df = df[df["layout"] == "Fruchterman-Reingold"]
    kk_df = df[df["layout"] == "Kamada-Kawai"]
    sugi_df = df[df["layout"] == "Sugiyama"]
    hola_df = df[df["layout"] == "HOLA"]
    circ_df = df[df["layout"] == "Circular"]
    dr_df = df[df["layout"] == "DRGraph"]
    ran_df = df[df["layout"] == "Random"]
    all_df = df[df["layout"] != "Random"]

    

    # Define the layouts and their corresponding dataframes
    dfs = {
        "Fruchterman-Reingold": fr_df,
        "Kamada-Kawai": kk_df,
        "DRGraph": dr_df,
        "Sugiyama": sugi_df,
        "HOLA": hola_df,
        "Circular": circ_df,
        "Random": ran_df,
        "All": all_df
    }

    layout_abr = {
        "Fruchterman-Reingold": "FR",
        "Kamada-Kawai": "KK",
        "Sugiyama": "Sugi",
        "HOLA": "HOLA",
        "Circular": "Circ",
        "DRGraph": "DRG",
        "Random": "Ran",
        "All": "All (excl. Ran)"
    }

    # Define the labels
    labels = ['angular_resolution', 'aspect_ratio', 'crossing_angle', 'edge_crossings', 'edge_length',
            'edge_orthogonality', 'gabriel_ratio', 'neighbourhood_preservation', 'node_resolution',
            'node_uniformity']


    # values = [["" for _ in range(len(dfs) + 1)] for _ in range(len(labels)+1)]

    
    # #print(values)
    # for i, label in enumerate(labels):
    #     values[i + 1][0] = label  # Set the label in the first column

    #     for j, (layout, df) in enumerate(dfs.items()):
    #         values[0][j + 1] = layout  # Set the layout name in the first row
    #         quartiles = np.percentile(df[label], [25, 50, 75])  # Calculate quartiles
    #         values[i + 1][j + 1] = [round(q, 3) for q in quartiles]  # Round quartiles to 3 decimal places and assign to values matrix


    # #print(values)
    # for row in values:
    #     print(row)


    quartile_df = pd.DataFrame(columns=["Label", "Layout", "First Quartile", "Median", "Third Quartile"])

    for label in labels:
        for layout, df in dfs.items():
            if layout == "All":
                quartiles = np.percentile(df[label], [25, 50, 75])  # Calculate quartiles
                quartile_df = quartile_df.append({
                    "Label": label,
                    "Layout": layout,
                    "First Quartile": round(quartiles[0], 3),
                    "Median": round(quartiles[1], 3),
                    "Third Quartile": round(quartiles[2], 3)
                }, ignore_index=True)

    # Display the quartile dataframe
    print(quartile_df)

    result_df = df.copy()

    label_medians = df[labels].median()

    # Iterate over each row in the dataframe
    for index, row in result_df.iterrows():
        print(index)
        values = row[labels]  # Get the values for the labels columns in the current row

        # Count the number of values that fall between the 1st and 3rd quartile
        num_within_quartiles = sum(np.percentile(values, [25, 75])[0] <= value <= np.percentile(values, [25, 75])[1] for value in values)

        # Assign the count to the new column 'num_within_range'
        result_df.loc[index, 'num_within_range'] = num_within_quartiles

        abs_diff_sum = np.sum(np.abs(values - label_medians))

        best_diff_sum = np.sum(1 - values)

        # Assign the sum to the new column 'abs_diff_sum'
        result_df.loc[index, 'abs_diff_sum'] = abs_diff_sum
        result_df.loc[index, 'best_diff_sum'] = best_diff_sum

    
    result_df.to_csv("GDM_with_diff.csv")

    # Find the maximum value of num_within_range
    # max_num_within_range = result_df['num_within_range'].max()

    # # Filter the dataframe to include only the rows with the maximum value
    # max_rows = result_df[result_df['num_within_range'] == max_num_within_range]

    # # Print the rows with the maximum value
    # print(max_rows)

    # # Save the resulting dataframe to a file
    # result_df.to_csv('result_dataframe.csv', index=False)

    # lowest_abs_diff_sum_rows = result_df.nsmallest(10, 'abs_diff_sum')

    # # Extract the filenames with the lowest abs_diff_sum
    # filenames_lowest_abs_diff_sum = lowest_abs_diff_sum_rows['filename'].values

    # # Print the filenames with the lowest abs_diff_sum
    # for filename in filenames_lowest_abs_diff_sum:
    #     print(filename)

    # import csv

    # # Specify the output file name
    # output_file = "numeric.csv"

    # # Open the output file in write mode
    # with open(output_file, mode='w', newline='') as file:
    #     writer = csv.writer(file)

    #     # Write each row of the values matrix to the CSV file
    #     for row in values:
    #         writer.writerow(row)


def get_typical(filename):

    df = pd.read_csv(filename)
    df = df.set_index("filename")

    df = df[df["layout"] != "Random"]

    df = df.drop(columns=['generator', 'layout', 'n', 'm', 'num_crossings', 'stress', 'centred_edge_crossings'])

    labels = ['angular_resolution', 'aspect_ratio', 'crossing_angle', 'edge_crossings', 'edge_length', 'edge_orthogonality', 'gabriel_ratio', 
              'neighbourhood_preservation', 'node_resolution', 'node_uniformity']
    
    medians = []
    for label in labels:
        med = df[label].median()
        print(f"{label} median: {med}")
        medians.append(med)

    total = 0
    i = 0
    for m in medians:
        total += m
        i += 1

    print(f"Mean of medians: {total/i}")

    graphs = [
        "DRGraph_j0_NWS_i12_n30_m44_p0.500_k2.graphml",
        "fruchterman-reingold_j0_NWS_i42_n60_m93_p0.500_k2.graphml",
        "kamada-kawai_j0_ER_i39_n90_m155_p0.039.graphml",
        "HOLA_j0_grafo1184.25.gml",
        "kamada-kawai_j0_g.52.1.graphml.xml",
        "fruchterman-reingold_j0_NWS_i21_n100_m144_p0.500_k3.graphml",
    ]

    
    row_avg = df.mean(axis=1)
    df['RowAverage'] = row_avg

    for graph in graphs:
        print(graph, df.loc[graph]['RowAverage'])


def main():

    #filename = "..\\Data\\metric_example2.csv"
    filename = "Graph_Drawing_Metrics.csv"
    #filename = "..\\Data\\GDM_final2.csv"

    get_typical(filename)
    #typical()
    #numeric_distribution(filename)



if __name__ == "__main__":

    main()
    