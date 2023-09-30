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
import seaborn as sns

def get_distributions(filename):
    df = pd.read_csv(filename)
    #df = df.drop(columns=['filename', 'SYM', 'CA', 'time'])
    df = df.drop(columns=['filename', 'type'])

    # for col in df:
        
    #     df.hist(column=col, bins=40, figsize=(10,8))
    #     plt.ylim(0,1446)
        
    #     plt.show()
        
        # df.hist(column=col, bins=40)
    

    fig, axs = plt.subplots(ncols=2, nrows=4)
    #hist = df.hist(bins=20)
    #hist1 = df.hist(bins=20, sharex=True)
    hist2 = df.hist(bins=40, sharex=True, sharey=True, ax=axs)
    

    plt.subplots_adjust(left=0.3,right=0.7,bottom=0.03,top=0.97,wspace=0.25,hspace=0.25)

    plt.show()


def correlation_matrix(filename):
    df = pd.read_csv(filename)
    df = df.set_index("filename")

    # df = df[~df.iloc[:, 1:].duplicated(keep='first')]
    df_no_hola_circ = df[~df['layout'].isin(["HOLA", "Circular"])]
    # print(df_no_hola_circ.head())
    df_hola_circ = df[df['layout'].isin(["HOLA", "Circular"])]
    #print(len(df_hola_circ))

    df_no_duplicates = df_no_hola_circ[~df_no_hola_circ.iloc[:, 1:].duplicated(keep='first')]

    duplicate_rows = df_no_hola_circ[df_no_hola_circ.iloc[:, 1:].duplicated(keep='first')]

    # Get the count of duplicate rows
    num_duplicate_rows = len(duplicate_rows)

    #print(f"Num duplicates: {num_duplicate_rows}")

    df = pd.concat([df_hola_circ, df_no_duplicates])

    fr_df = df[df["layout"] == "Fruchterman-Reingold"]
    kk_df = df[df["layout"] == "Kamada-Kawai"]
    hola_df = df[df["layout"] == "HOLA"]
    sugi_df = df[df["layout"] == "Sugiyama"]
    dr_df = df[df["layout"] == "DRGraph"]
    ran_df = df[df["layout"] == "Random"]
    all_df = df[df["layout"] != "Random"]


    dfs = {"Fruchterman-Reingold":fr_df, "Kamada-Kawai":kk_df ,"DRGraph":dr_df, "Sugiyama":sugi_df, "HOLA":hola_df, "Random":ran_df, "All (excl. Random)":all_df}
    dfs = {"All (excl. Random)":all_df}


    labels = ['angular_resolution', 'aspect_ratio', 'crossing_angle', 'edge_crossings', 'edge_length', 'edge_orthogonality', 'gabriel_ratio', 
              'neighbourhood_preservation', 'node_resolution', 'node_uniformity']

    # labels_proper = {"ar":"Angular Resolution", "asp":"Aspect Ratio", "ca":"Crossing Angle", "ec":"Edge Crossings", 
    #                  "el":"Edge Lengths", "eo":"Edge Orthogonality", "gr":"Gabriel Ratio", "np":"Neighbourhood Preservation", 
    #                  "nr":"Node Resolution", "nu":"Node Unifortmity"}

    labels_proper = {'angular_resolution':'Angular Resolution', 'aspect_ratio':'Aspect Ratio', 'crossing_angle':'Crossing Angle', 
                    'edge_crossings':'Edge Crossings', 'edge_length':'Edge Lengths', 'edge_orthogonality':'Edge Orthogonality', 
                    'gabriel_ratio':'Gabriel Ratio', 'neighbourhood_preservation':'Neighbourhood Preservation', 
                    'node_resolution':'Node Resolution', 'node_uniformity':'Node Unifortmity'}
    
    labels_abr = {'angular_resolution':'AR', 'aspect_ratio':'Asp', 'crossing_angle':'CA', 
                    'edge_crossings':'EC', 'edge_length':'EL', 'edge_orthogonality':'EO', 
                    'gabriel_ratio':'GR', 'neighbourhood_preservation':'NP', 
                    'node_resolution':'NR', 'node_uniformity':'NU',}
    
    for k, v in dfs.items():

        v = v.drop(columns=['generator', 'layout', 'n', 'm', 'num_crossings', 'stress', 'centred_edge_crossings'])

        corr_matrix = v[labels].corr() #pearsons

        corr_matrix = round(corr_matrix, 3)


        mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
        corr_matrix = corr_matrix.mask(mask)

        cmap = plt.get_cmap('PuOr').copy()
        cmap.set_bad(color='white')
        fig, ax = plt.subplots(figsize=(16.5, 11.7))
        im = ax.imshow(corr_matrix, cmap=cmap)
        im.set_clim(-1, 1)
        ax.grid(False)
        ax.xaxis.set(ticks=range(len(labels)), ticklabels=[labels_abr[label] for label in labels])
        ax.yaxis.set(ticks=range(len(labels)), ticklabels=[labels_abr[label] for label in labels])
        ax.set_title(k, fontsize=14)

        for label in ax.get_xticklabels():
            label.set_fontsize(14)

        for label in ax.get_yticklabels():
            label.set_fontsize(14)

        for i in range(len(labels)):
            for j in range(len(labels)):
                if i >= j:
                    ax.text(j, i, str(corr_matrix.iloc[i,j]), ha='center', va='center', color='black', fontweight='bold', fontsize=14)
                pass
        cbar = ax.figure.colorbar(im, ax=ax, format='% .2f')
        cbar.ax.tick_params(labelsize=14)
        plt.savefig(k + "_corr.pdf", format="pdf", bbox_inches="tight")
        plt.show()


def correlations_same_fig(filename):
    df = pd.read_csv(filename)
    df = df.set_index("filename")

    # df = df[~df.iloc[:, 1:].duplicated(keep='first')]
    df_no_hola_circ = df[~df['layout'].isin(["HOLA", "Circular"])]
    # print(df_no_hola_circ.head())
    df_hola_circ = df[df['layout'].isin(["HOLA", "Circular"])]
    #print(len(df_hola_circ))

    df_no_duplicates = df_no_hola_circ[~df_no_hola_circ.iloc[:, 1:].duplicated(keep='first')]

    duplicate_rows = df_no_hola_circ[df_no_hola_circ.iloc[:, 1:].duplicated(keep='first')]

    # Get the count of duplicate rows
    num_duplicate_rows = len(duplicate_rows)

    #print(f"Num duplicates: {num_duplicate_rows}")

    df = pd.concat([df_hola_circ, df_no_duplicates])

    fr_df = df[df["layout"] == "Fruchterman-Reingold"]
    kk_df = df[df["layout"] == "Kamada-Kawai"]
    dr_df = df[df["layout"] == "DRGraph"]
    sugi_df = df[df["layout"] == "Sugiyama"]
    hola_df = df[df["layout"] == "HOLA"]
    circ_df = df[df["layout"] == "Circular"]
    ran_df = df[df["layout"] == "Random"]
    all_df = df[df["layout"] != "Random"]

    dfs = {
        "Fruchterman-Reingold": fr_df,
        "Kamada-Kawai": kk_df,
        "DRGraph": dr_df,
        "Sugiyama": sugi_df,
        "HOLA": hola_df,
        "Circular": circ_df,
        "Random": ran_df,
        "All (excl. Random)": all_df
    }


    dfs1 = {
        "Fruchterman-Reingold": fr_df,
        "Kamada-Kawai": kk_df,
        "DRGraph": dr_df,
        "Sugiyama": sugi_df,
    }

    dfs2 = {
        "HOLA": hola_df,
        "Circular": circ_df,
        "Random": ran_df,
        "All (excl. Random)": all_df
    }
    

    labels = [
        'angular_resolution', 'aspect_ratio', 'crossing_angle', 'edge_crossings',
        'edge_length', 'edge_orthogonality', 'gabriel_ratio',
        'neighbourhood_preservation', 'node_resolution',
        'node_uniformity', #'stress'
    ]

    labels_abr = {
        'angular_resolution': 'AR',
        'aspect_ratio': 'Asp',
        'crossing_angle': 'CA',
        'edge_crossings': 'EC',
        'edge_length': 'EL',
        'edge_orthogonality': 'EO',
        'gabriel_ratio': 'GR',
        'neighbourhood_preservation': 'NP',
        'node_resolution': 'NR',
        'node_uniformity': 'NU',
        # 'stress': 'Str'
    }

    #fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig, axes = plt.subplots(2, 2, figsize=(9.5, 9), gridspec_kw={'hspace': 0.175})

    for (k, v), ax in zip(dfs1.items(), axes.flatten()):
        v = v.drop(columns=['generator', 'layout', 'n', 'm', 'num_crossings', 'stress', 'centred_edge_crossings'])

        corr_matrix = v[labels].corr()
        corr_matrix = round(corr_matrix, 3)

        #print(corr_matrix)

        mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
        corr_matrix = corr_matrix.mask(mask)

        cmap = plt.get_cmap('PuOr').copy()
        cmap.set_bad(color='white')

        im = ax.imshow(corr_matrix, cmap=cmap)
        im.set_clim(-1, 1)
        ax.grid(False)
        ax.xaxis.set(ticks=range(len(labels)), ticklabels=[labels_abr[label] for label in labels])
        ax.yaxis.set(ticks=range(len(labels)), ticklabels=[labels_abr[label] for label in labels])
        ax.set_title(k)

        ax.margins(0.01)

        for i in range(len(labels)):
            for j in range(len(labels)):
                if i >= j:
                    ax.text(j, i, str(corr_matrix.iloc[i, j]), ha='center', va='center', color='black', size=6, fontweight='bold')

        ax.margins(0.05)  # Reduce the padding around each matrix

    #cbar = fig.colorbar(im, ax=axes.ravel().tolist(), format='%.2f')
    #cbar = fig.colorbar(im, ax=axes.ravel().tolist(), format='%.2f', shrink=0.5)

    #cbar.ax.tick_params(labelsize=6)

    #plt.tight_layout(pad=0.5)  # Adjust the overall padding between subplots
    #plt.tight_layout()
    #plt.subplots_adjust(wspace=0.03, hspace=0.2)

    cbar_ax = fig.add_axes([0.95, 0.1, 0.02, 0.8])  # Adjust the position and size of the colorbar axis as needed
    cbar = fig.colorbar(im, cax=cbar_ax, format='%.2f')
    cbar.ax.tick_params(labelsize=6)

    plt.savefig("corr_all_1.pdf", format="pdf", bbox_inches="tight")
    plt.show()

    #############################################################################################################

    fig, axes = plt.subplots(2, 2, figsize=(9.5, 9), gridspec_kw={'hspace': 0.175})

    for (k, v), ax in zip(dfs2.items(), axes.flatten()):
        v = v.drop(columns=['generator', 'layout', 'n', 'm', 'num_crossings', 'stress', 'centred_edge_crossings'])

        corr_matrix = v[labels].corr()
        corr_matrix = round(corr_matrix, 3)

        #print(corr_matrix)

        mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
        corr_matrix = corr_matrix.mask(mask)

        cmap = plt.get_cmap('PuOr').copy()
        cmap.set_bad(color='white')

        im = ax.imshow(corr_matrix, cmap=cmap)
        im.set_clim(-1, 1)
        ax.grid(False)
        ax.xaxis.set(ticks=range(len(labels)), ticklabels=[labels_abr[label] for label in labels])
        ax.yaxis.set(ticks=range(len(labels)), ticklabels=[labels_abr[label] for label in labels])
        ax.set_title(k)

        ax.margins(0.01)

        for i in range(len(labels)):
            for j in range(len(labels)):
                if i >= j:
                    ax.text(j, i, str(corr_matrix.iloc[i, j]), ha='center', va='center', color='black', size=6, fontweight='bold')

        ax.margins(0.05)  # Reduce the padding around each matrix

    #cbar = fig.colorbar(im, ax=axes.ravel().tolist(), format='%.2f')
    #cbar = fig.colorbar(im, ax=axes.ravel().tolist(), format='%.2f', shrink=0.5)

    #cbar.ax.tick_params(labelsize=6)

    #plt.tight_layout(pad=0.5)  # Adjust the overall padding between subplots
    #plt.tight_layout()
    #plt.subplots_adjust(wspace=0.03, hspace=0.2)

    cbar_ax = fig.add_axes([0.95, 0.1, 0.02, 0.8])  # Adjust the position and size of the colorbar axis as needed
    cbar = fig.colorbar(im, cax=cbar_ax, format='%.2f')
    cbar.ax.tick_params(labelsize=6)

    plt.savefig("corr_all_2.pdf", format="pdf", bbox_inches="tight")
    plt.show()




def corr_matrix_sep(filename):
        # Get the correlation matrix
    df = pd.read_csv(filename)
    df = df.drop(columns=['filename'])
    #df = df.drop(columns=['NO'])

    bba_df = df[df["type"] == "BBA"]
    er_df = df[df["type"] == "ER"]
    nws_df = df[df["type"] == "NWS"]
    lfr_df = df[df["type"] == "LFR"]

    dfs = {"BBA":bba_df, "ER":er_df, "NWS":nws_df, "LFR":lfr_df, "All":df}

    for x in dfs.values():
        x.drop(columns=['type'])


    #df = df[:-2]

    labels = ["n", "m", "density", "diameter", "acc", "ascc", "apl", "sr"]


    print(len(df))
    i = 0
    j = 0
    fullfig, fullax = plt.subplots(2, 3)
    for v in dfs.values():

        corr_matrix = v.corr() #pearsons
        corr_matrix = round(corr_matrix, 3)

        for i, l in zip(range(len(labels)), labels):
            for j, l2 in zip(range(len(labels)), labels):
                if i > j:
                    corr_matrix[l][l2] = None

        print(corr_matrix)

        cmap = plt.get_cmap('plasma').copy()
        cmap.set_bad(color='white')
        fig, ax = plt.subplots()
        im = ax.imshow(corr_matrix, cmap=cmap)
        im.set_clim(-1, 1)
        ax.grid(False)
        ax.xaxis.set(ticks=range(len(labels)), ticklabels=labels)
        ax.yaxis.set(ticks=range(len(labels)), ticklabels=labels)

        for i in range(len(labels)):
            for j in range(len(labels)):
                if i >= j:
                    ax.text(j, i, str(corr_matrix.iloc[i,j]), ha='center', va='center', color='black')
                pass
        cbar = ax.figure.colorbar(im, ax=ax, format='% .2f')
        #plt.savefig("correlations.pdf", format="pdf", bbox_inches="tight")
        plt.show()

def corr_matrix_sep2(filename):
    df = pd.read_csv(filename)
    df = df.set_index("filename")

    fr_df = df[df["layout"] == "Fruchterman-Reingold"]
    kk_df = df[df["layout"] == "Kamada-Kawai"]
    #hola_df = df[df["layout"] == ""]
    sugi_df = df[df["layout"] == "Sugiyama"]
    dr_df = df[df["layout"] == "DRGraph"]
    ran_df = df[df["layout"] == "Random"]
    all_df = df.copy()

    #dfs = {"FR":fr_df, "KK":kk_df, "Sugi":sugi_df, "DR":dr_df, "Ran":ran_df, "All":all_df}
    #dfs = {"FR":fr_df, "KK":kk_df, "Sugi":sugi_df, "DR":dr_df, "All":all_df}
    dfs = {"Fruchterman-Reingold":fr_df, "Kamada-Kawai":kk_df, "Sugiyama":sugi_df, "DRGraph":dr_df, "Random":ran_df, "All":all_df}


    #labels = ["ar", "asp", "ca", "ec", "el", "eo", "gr", "np", "nr", "nu"]
    labels = ['angular_resolution', 'aspect_ratio', 'crossing_angle', 'edge_crossings', 'edge_length', 'edge_orthogonality', 'gabriel_ratio', 
              'neighbourhood_preservation', 'node_resolution', 'node_uniformity', 'stress']

    # labels_proper = {"ar":"Angular Resolution", "asp":"Aspect Ratio", "ca":"Crossing Angle", "ec":"Edge Crossings", 
    #                  "el":"Edge Lengths", "eo":"Edge Orthogonality", "gr":"Gabriel Ratio", "np":"Neighbourhood Preservation", 
    #                  "nr":"Node Resolution", "nu":"Node Unifortmity"}

    labels_proper = {'angular_resolution':'Angular Resolution', 'aspect_ratio':'Aspect Ratio', 'crossing_angle':'Crossing Angle', 
                    'edge_crossings':'Edge Crossings', 'edge_length':'Edge Lengths', 'edge_orthogonality':'Edge Orthogonality', 
                    'gabriel_ratio':'Gabriel Ratio', 'neighbourhood_preservation':'Neighbourhood Preservation', 
                    'node_resolution':'Node Resolution', 'node_uniformity':'Node Unifortmity', 'stress':'Stress'}
    
    labels_abr = {'angular_resolution':'AR', 'aspect_ratio':'Asp', 'crossing_angle':'CA', 
                    'edge_crossings':'EC', 'edge_length':'EL', 'edge_orthogonality':'EO', 
                    'gabriel_ratio':'GR', 'neighbourhood_preservation':'NP', 
                    'node_resolution':'NR', 'node_uniformity':'NU', 'stress':'Str'}


    corr_matrices = []

    for k, v in dfs.items():
        #print(v)
        v = v.drop(columns=['generator', 'layout', 'n', 'm', 'num_crossings'])

        corr_matrix = v.corr() #pearsons
        corr_matrix = round(corr_matrix, 3)

        for i, l in zip(range(len(labels)), labels):
            for j, l2 in zip(range(len(labels)), labels):
                if i > j:
                    corr_matrix[l][l2] = None

        corr_matrices.append((k, corr_matrix))

        
    fig, axes = plt.subplots(nrows=2, ncols=3)

    
    cmap = plt.get_cmap('plasma').copy()
    cmap.set_bad(color='white')

    # Display each correlation matrix in a subplot
    for i, ax in enumerate(axes.flat):
        if i == 5:
            continue
        corr_matrix = corr_matrices[i][1]
        corr_matrix = round(corr_matrix, 3)
        im = ax.imshow(corr_matrix, cmap=cmap)
        im.set_clim(-1, 1)
        ax.grid(False)
        ax.xaxis.set(ticks=range(len(corr_matrix)), ticklabels=[x if len(x) <= 3 else x[0:4] for x in labels])
        ax.yaxis.set(ticks=range(len(corr_matrix)), ticklabels=[x if len(x) <= 3 else x[0:4] for x in labels])
        for j in range(len(corr_matrix)):
            for k in range(len(corr_matrix)):
                if j >= k:
                    ax.text(k, j, str(corr_matrix.iloc[j,k]), ha='center', va='center', color='black')
        ax.set_title(corr_matrices[i][0])


    # Add a colorbar
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), format='% .2f')
    #plt.savefig("correlations.pdf", format="pdf")
    plt.show()
    


def get_distributions_sep(filename, bins=50):

    df = pd.read_csv(filename)
    df = df.drop(columns=['filename'])
    #df = df.drop(columns=['NO'])

    bba_df = df[df["type"] == "BBA"]
    er_df = df[df["type"] == "ER"]
    nws_df = df[df["type"] == "NWS"]
    lfr_df = df[df["type"] == "LFR"]

    dfs = {"BBA":bba_df, "ER":er_df, "NWS":nws_df, "LFR":lfr_df, "All":df}
    #df = df[:-2]

    labels = ["n", "m", "density", "diameter", "acc", "ascc", "apl", "sr"]

    df_list = []

    for k, v in dfs.items():
        v = v.drop(columns=['type'])
        df_list.append((k, v))

    

    for l in labels:
        fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(10, 8))
        fig.suptitle(f"Bins: {bins}")
        # Plot the histograms of each dataframe on a different subplot
        axs_list = axs.ravel()
        for i, dfx in enumerate(df_list):
            ax = axs_list[i]
            ax.hist(dfx[1][l], bins=bins)
            ax.set_title(dfx[0])
            ax.set_xlabel(l)
            ax.set_ylabel('Frequency')

        # Adjust the layout and display the plot
        plt.tight_layout()
        plt.show()





def violin_plots_individual(filename, df_label="BBA", highlight=None):
    df = pd.read_csv(filename)
    df = df.set_index("filename")

    fr_df = df[df["layout"] == "Fruchterman-Reingold"]
    kk_df = df[df["layout"] == "Kamada-Kawai"]
    hola_df = df[df["layout"] == "HOLA"]
    sugi_df = df[df["layout"] == "Sugiyama"]
    dr_df = df[df["layout"] == "DRGraph"]
    ran_df = df[df["layout"] == "Random"]
    all_df = df[df["layout"] != "Random"]

    #dfs = {"FR":fr_df, "KK":kk_df, "Sugi":sugi_df, "DR":dr_df, "Ran":ran_df, "All":all_df}
    #dfs = {"FR":fr_df, "KK":kk_df, "Sugi":sugi_df, "DR":dr_df, "All":all_df}
    dfs = {"HOLA": hola_df, "Fruchterman-Reingold":fr_df, "Kamada-Kawai":kk_df, "Sugiyama":sugi_df, "DRGraph":dr_df, "Random":ran_df, "All (excl. Random)":all_df}


    #df = dfs[df_label]
    #print(df)
    #legend = "filename,type,n,m,ad,density,diameter,acc,ascc,apl,sr\n"
    #df = df.set_index("filename")


    #labels = ["ar", "asp", "ca", "ec", "el", "eo", "gr", "np", "nr", "nu"]
    labels = ['angular_resolution', 'aspect_ratio', 'crossing_angle', 'edge_crossings', 'centred_edge_crossings', 'edge_length', 'edge_orthogonality', 'gabriel_ratio', 
              'neighbourhood_preservation', 'node_resolution', 'node_uniformity', 'stress']

    # labels_proper = {"ar":"Angular Resolution", "asp":"Aspect Ratio", "ca":"Crossing Angle", "ec":"Edge Crossings", 
    #                  "el":"Edge Lengths", "eo":"Edge Orthogonality", "gr":"Gabriel Ratio", "np":"Neighbourhood Preservation", 
    #                  "nr":"Node Resolution", "nu":"Node Unifortmity"}

    labels_proper = {'angular_resolution':'Angular Resolution', 'aspect_ratio':'Aspect Ratio', 'crossing_angle':'Crossing Angle', 
                    'edge_crossings':'Edge Crossings', 'centred_edge_crossings': 'Edge Crossings (C)', 'edge_length':'Edge Lengths', 'edge_orthogonality':'Edge Orthogonality', 
                    'gabriel_ratio':'Gabriel Ratio', 'neighbourhood_preservation':'Neighbourhood Preservation', 
                    'node_resolution':'Node Resolution', 'node_uniformity':'Node Unifortmity', 'stress':'Stress'}
    
    labels_abr = {'angular_resolution':'AR', 'aspect_ratio':'Asp', 'crossing_angle':'CA', 
                    'edge_crossings':'EC', 'centred_edge_crossings': 'EC(C)', 'edge_length':'EL', 'edge_orthogonality':'EO', 
                    'gabriel_ratio':'GR', 'neighbourhood_preservation':'NP', 
                    'node_resolution':'NR', 'node_uniformity':'NU', 'stress':'Str'}


    for k, v in dfs.items():
        v = v.drop(columns=['generator', 'layout', 'n', 'm', 'num_crossings'])
        df_list = []
        for label in labels:
            data = v[label].dropna().to_numpy()
            df_list.append(data)

        fig, ax = plt.subplots()
        ax.violinplot(df_list, showmedians=True, vert=False)
        # ax.set_xticks(range(1, len(labels) + 1))
        # ax.set_xticklabels([labels_abr[label] for label in labels])
        ax.set_yticks(range(1, len(labels) + 1))
        ax.set_yticklabels([labels_abr[label] for label in labels])
        ax.set_title(k)
        plt.show()

    # df_list = []


    # for k, v in dfs.items():
    #     v = v.drop(columns=['generator', 'layout', 'n', 'm', 'num_crossings', 'stress'])
    #     df_list.append((k, v))


    # for label in labels:
    #     #print(label)
    #     data = [dfx[label] for _, dfx in df_list]
    #     #data = [df[label].dropna().to_numpy() for _, df in df_list]

    #     fig, ax = plt.subplots()
    #     ax.violinplot(data)
    #     ax.set_xticks([1, 2, 3, 4, 5])
    #     ax.set_xticklabels(dfs.keys())
    #     ax.set_title(label)
    #     plt.show()

    # for label in labels:
    #     data = [df[label] for _, df in df_list]
    #     fig, ax = plt.subplots()
    #     parts = ax.violinplot(data, showmeans=False, showmedians=False, showextrema=True)

    #     ax.set_xticks(np.arange(len(dfs)) + 1)
    #     ax.set_xticklabels(dfs.keys())
    #     ax.set_title(labels_proper[label])

    #     type_index = {v:k for (k,v) in enumerate(dfs.keys())}

    #     # Add end point labels
    #     for i, side in enumerate(parts['bodies']):
    #         min_val = np.min(side.get_paths()[0].vertices[:, 1])
    #         max_val = np.max(side.get_paths()[0].vertices[:, 1])
    #         ax.text(i+1, min_val, f"{min_val:.2f}")
    #         ax.text(i+1, max_val, f"{max_val:.2f}")

    #     if highlight is not None:
    #         ax.scatter(type_index[df.loc[highlight]["type"]] + 1, df.loc[highlight][label], color='r', marker='o')

    #         ax.scatter(len(dfs), df.loc[highlight][label], color='r', marker='o')

    #         legend_labels = [f"{highlight}: {label} = {df.loc[highlight][label]}"]
    #         legend_handles = [plt.scatter([], [], color='red', marker='o', label=label) for i, label in enumerate(legend_labels)]
    #         ax.legend(handles=legend_handles, labels=legend_labels, loc="upper center", bbox_to_anchor=(0.5, 1.17))


    #     plt.show()


def violin_one_fig(filename):
    df = pd.read_csv(filename)
    df = df.set_index("filename")

    # Create dataframes for each layout
    fr_df = df[df["layout"] == "Fruchterman-Reingold"]
    kk_df = df[df["layout"] == "Kamada-Kawai"]
    sugi_df = df[df["layout"] == "Sugiyama"]
    dr_df = df[df["layout"] == "DRGraph"]
    ran_df = df[df["layout"] == "Random"]
    all_df = df.copy()

    print(min(all_df["node_uniformity"]))

    # Define the layouts and their corresponding dataframes
    dfs = {
        "Fruchterman-Reingold": fr_df,
        "Kamada-Kawai": kk_df,
        "Sugiyama": sugi_df,
        "DRGraph": dr_df,
        "Random": ran_df,
        "All": all_df
    }

    # Define the labels
    labels = ['angular_resolution', 'aspect_ratio', 'crossing_angle', 'edge_crossings', 'centred_edge_crossings', 'edge_length', 'edge_orthogonality', 'gabriel_ratio', 
              'neighbourhood_preservation', 'node_resolution', 'node_uniformity', 'stress']

    # labels_proper = {"ar":"Angular Resolution", "asp":"Aspect Ratio", "ca":"Crossing Angle", "ec":"Edge Crossings", 
    #                  "el":"Edge Lengths", "eo":"Edge Orthogonality", "gr":"Gabriel Ratio", "np":"Neighbourhood Preservation", 
    #                  "nr":"Node Resolution", "nu":"Node Unifortmity"}

    labels_proper = {'angular_resolution':'Angular Resolution', 'aspect_ratio':'Aspect Ratio', 'crossing_angle':'Crossing Angle', 
                    'edge_crossings':'Edge Crossings', 'centred_edge_crossings': 'Edge Crossings (C)', 'edge_length':'Edge Lengths', 'edge_orthogonality':'Edge Orthogonality', 
                    'gabriel_ratio':'Gabriel Ratio', 'neighbourhood_preservation':'Neighbourhood Preservation', 
                    'node_resolution':'Node Resolution', 'node_uniformity':'Node Unifortmity', 'stress':'Stress'}
    
    labels_abr = {'angular_resolution':'AR', 'aspect_ratio':'Asp', 'crossing_angle':'CA', 
                    'edge_crossings':'EC', 'centred_edge_crossings': 'EC(C)', 'edge_length':'EL', 'edge_orthogonality':'EO', 
                    'gabriel_ratio':'GR', 'neighbourhood_preservation':'NP', 
                    'node_resolution':'NR', 'node_uniformity':'NU', 'stress':'Str'}

    # Create subplots for each layout in one row
    fig, axes = plt.subplots(1, len(dfs), figsize=(16, 5))

    # Loop over the layouts
    for i, (layout, v) in enumerate(dfs.items()):
        v = v.drop(columns=['generator', 'layout', 'n', 'm', 'num_crossings'])
        df_list = []
        
        # Retrieve data for each label
        for label in labels:
            data = v[label].dropna().to_numpy()
            df_list.append(data)
        
        # Create a horizontal violin plot
        axes[i].violinplot(df_list, showmedians=True, vert=False)  # Set vert=False for horizontal plot
        axes[i].set_yticks(range(1, len(labels) + 1))
        axes[i].set_yticklabels([labels_abr[label] for label in labels])
        axes[i].set_title(layout)



    plt.tight_layout()
    plt.show()




def violin_one_fig_vert(filename):
    df = pd.read_csv(filename)
    df = df.set_index("filename")

    # Create dataframes for each layout
    fr_df = df[df["layout"] == "Fruchterman-Reingold"]
    kk_df = df[df["layout"] == "Kamada-Kawai"]
    sugi_df = df[df["layout"] == "Sugiyama"]
    dr_df = df[df["layout"] == "DRGraph"]
    ran_df = df[df["layout"] == "Random"]
    all_df = df.copy()

    # Define the layouts and their corresponding dataframes
    dfs = {
        "Fruchterman-Reingold": fr_df,
        "Kamada-Kawai": kk_df,
        "Sugiyama": sugi_df,
        "DRGraph": dr_df,
        "Random": ran_df,
        "All": all_df
    }

    layout_abr = {
        "Fruchterman-Reingold": "FR",
        "Kamada-Kawai": "KK",
        "Sugiyama": "Sugi",
        "DRGraph": "DRG",
        "Random": "Ran",
        "All": "All"
    }

    # Define the labels
    labels = ['angular_resolution', 'aspect_ratio', 'crossing_angle', 'edge_crossings', 'centred_edge_crossings', 'edge_length', 'edge_orthogonality', 'gabriel_ratio', 
              'neighbourhood_preservation', 'node_resolution', 'node_uniformity', 'stress']

    # labels_proper = {"ar":"Angular Resolution", "asp":"Aspect Ratio", "ca":"Crossing Angle", "ec":"Edge Crossings", 
    #                  "el":"Edge Lengths", "eo":"Edge Orthogonality", "gr":"Gabriel Ratio", "np":"Neighbourhood Preservation", 
    #                  "nr":"Node Resolution", "nu":"Node Unifortmity"}

    labels_proper = {'angular_resolution':'Angular Resolution', 'aspect_ratio':'Aspect Ratio', 'crossing_angle':'Crossing Angle', 
                    'edge_crossings':'Edge Crossings', 'centred_edge_crossings': 'Edge Crossings (C)', 'edge_length':'Edge Lengths', 'edge_orthogonality':'Edge Orthogonality', 
                    'gabriel_ratio':'Gabriel Ratio', 'neighbourhood_preservation':'Neighbourhood Preservation', 
                    'node_resolution':'Node Resolution', 'node_uniformity':'Node Unifortmity', 'stress':'Stress'}
    
    labels_abr = {'angular_resolution':'AR', 'aspect_ratio':'Asp', 'crossing_angle':'CA', 
                    'edge_crossings':'EC', 'centred_edge_crossings': 'EC(C)', 'edge_length':'EL', 'edge_orthogonality':'EO', 
                    'gabriel_ratio':'GR', 'neighbourhood_preservation':'NP', 
                    'node_resolution':'NR', 'node_uniformity':'NU', 'stress':'Str'}
    from matplotlib.ticker import FixedLocator

    # Create a figure with a grid of subplots
    fig, axes = plt.subplots(len(dfs), len(labels), figsize=(20, 30))

    # Loop over the layouts
    for i, (layout, v) in enumerate(dfs.items()):
        v = v.drop(columns=['generator', 'layout', 'n', 'm', 'num_crossings', 'stress'])

        # Retrieve data for each label
        for j, label in enumerate(labels):
            data = v[label].dropna().to_numpy()

            # Create a violin plot for the current layout and label
            ax = axes[i, j]
            ax.violinplot(data, showmedians=True, vert=False)  # Horizontal plot

            ax.set_xlim(0, 1)
            # Remove the titles on each axis
            ax.set_title('')

            # Rotate the x-axis labels horizontally
            ax.tick_params(axis='x', rotation=0)
            ax.tick_params(axis='y', length=0)  # Remove vertical tick marks

            # Show tick marks only for the outer right and bottom plots
            if i != len(dfs) - 1:
                ax.set_xticklabels([])
            if j != 0:
                ax.set_yticklabels([])

            # Set y-axis label as layout name
            if j == 0:
                ax.set_yticks([])
                ax.text(-0.1, 0.5, layout_abr[layout], va='center', ha='right', fontsize=10, transform=ax.transAxes)

            # # Set x-axis label
            # if j == 0 and i == 0:
            #     ax.set_xlabel(labels_abr[label])

            # Adjust x-axis tick alignment
            if i == len(dfs) - 1:
                ax.set_xticks([0, 0.5, 1])
                ax.set_xticklabels(['0', '0.5', '1'], ha='center')
            else:
                ax.set_xticks([])

    # Set titles for each column of violin plots in the top row
    for j, label in enumerate(labels):
        axes[0, j].set_title(labels_abr[label])

    # Remove empty subplots
    for i in range(len(dfs), len(axes)):
        for j in range(len(labels)):
            fig.delaxes(axes[i, j])

    # Adjust spacing
    fig.subplots_adjust(hspace=0.1)

    plt.tight_layout()
    plt.show()

def violin_one_fig_transpose_generator(filename):
    df = pd.read_csv(filename)
    df = df.set_index("filename")

    # Create dataframes for each layout
    fr_df = df[df["layout"] == "Fruchterman-Reingold"]
    kk_df = df[df["layout"] == "Kamada-Kawai"]
    sugi_df = df[df["layout"] == "Sugiyama"]
    hola_df = df[df["layout"] == "HOLA"]
    dr_df = df[df["layout"] == "DRGraph"]
    ran_df = df[df["layout"] == "Random"]
    all_df = df[df["layout"] != "Random"]


    bba_df = df[df["generator"] == "Barabasi-Albert"]
    er_df = df[df["generator"] == "Erdos-Renyi"]
    geo_df = df[df["generator"] == "Geometric"]
    lfr_df = df[df["generator"] == "Lancichinetti-Fortunato-Radicchi"]
    nws_df = df[df["generator"] == "Newman-Watts-Strogatz"]
    north_df = df[df["generator"] == "North"]
    rome_df = df[df["generator"] == "Rome"]
    sbm_df = df[df["generator"] == "Stochastic-Block-Model"]
    all_gen_df = df.copy()

    
    # Define the layouts and their corresponding dataframes
    # dfs = {
    #     "Fruchterman-Reingold": fr_df,
    #     "Kamada-Kawai": kk_df,
    #     "Sugiyama": sugi_df,
    #     "HOLA": hola_df,
    #     "DRGraph": dr_df,
    #     "Random": ran_df,
    #     "All": all_df
    # }
    dfs = {
        "BBA": bba_df,
        "ER": er_df,
        "GEO": geo_df,
        "LFR": lfr_df,
        "NWS": nws_df,
        "SBM": sbm_df,
        "North": north_df,
        "Rome": rome_df,
        "All": all_gen_df
    }


    # layout_abr = {
    #     "Fruchterman-Reingold": "FR",
    #     "Kamada-Kawai": "KK",
    #     "Sugiyama": "Sugi",
    #     "HOLA": "HOLA",
    #     "DRGraph": "DRG",
    #     "Random": "Ran",
    #     "All": "All (excl. Ran)"
    # }

    layout_abr = {
        "BBA": "BBA",
        "ER": "ER",
        "GEO": "GEO",
        "LFR": "LFR",
        "NWS": "NWS",
        "SBM": "SBM",
        "North": "North",
        "Rome": "Rome",
        "All": "All"
    }

    # Define the labels
    labels = ['angular_resolution', 'aspect_ratio', 'crossing_angle', 'edge_crossings', 'edge_length',
            'edge_orthogonality', 'gabriel_ratio', 'neighbourhood_preservation', 'node_resolution',
            'node_uniformity', 'stress']
    
    #labels = ['stress_not_normal']

    # Define the proper labels and abbreviations
    labels_proper = {
        'angular_resolution': 'Angular Resolution',
        'aspect_ratio': 'Aspect Ratio',
        'crossing_angle': 'Crossing Angle',
        'edge_crossings': 'Edge Crossings',
        'edge_length': 'Edge Lengths',
        'edge_orthogonality': 'Edge Orthogonality',
        'gabriel_ratio': 'Gabriel Ratio',
        'neighbourhood_preservation': 'Neighbourhood Preservation',
        'node_resolution': 'Node Resolution',
        'node_uniformity': 'Node Unifortmity',
        'stress': 'Stress'
    }

    labels_abr = {
        'angular_resolution': 'AR',
        'aspect_ratio': 'Asp',
        'crossing_angle': 'CA',
        'edge_crossings': 'EC',
        'edge_length': 'EL',
        'edge_orthogonality': 'EO',
        'gabriel_ratio': 'GR',
        'neighbourhood_preservation': 'NP',
        'node_resolution': 'NR',
        'node_uniformity': 'NU',
        'stress': 'Str'
    }

    # labels_abr = {
    #     'stress_not_normal':'StrNN'
    # }
    from matplotlib.ticker import FixedLocator

    fig, axes = plt.subplots(len(labels), len(dfs), figsize=(16.5, 11.7))

    # Loop over the layouts
    for i, (layout, v) in enumerate(dfs.items()):
        v = v.drop(columns=['generator', 'layout', 'n', 'm', 'num_crossings'])
        #print(layout)

        # Retrieve data for each label
        for j, label in enumerate(labels):
            #print(label)
            data = v[label].dropna().to_numpy()

            # Create a violin plot for the current layout and label
            ax = axes[j, i]  # Flipped indexing
            #ax.violinplot(data, showmedians=True)  # Horizontal plot

            vp = ax.violinplot(data, showmedians=True)  # Horizontal plot

            # Adjust properties of the lines
            vp['cmedians'].set_linewidth(0.6)  # Fainter median line
            vp['cmins'].set_linewidth(0.6)  # Fainter minimum line
            vp['cmaxes'].set_linewidth(0.6)  # Fainter maximum line
            vp['cbars'].set_linewidth(0.6)  # Fainter quartile lines

            ax.set_ylim(0, 1)
            # Remove the titles on each axis
            ax.set_title('')

            # Rotate the x-axis labels horizontally
            ax.tick_params(axis='y', rotation=0)
            ax.tick_params(axis='x', length=0)  # Remove vertical tick marks

            # Show tick marks only for the outer right and bottom plots
            ax.set_yticklabels([], ha='left')  # Align left
            ax.set_xticklabels([], ha='center')  # Align center

            # Set y-axis label as layout name
            if i == 0:
                ax.set_yticks([])
                ax.text(-0.025, 0.5, labels_abr[label], va='center', ha='right', fontsize=10, transform=ax.transAxes)

            ax.set_xticks([])
            # Adjust x-axis tick alignment
            if i == len(dfs) - 1:
                ax.yaxis.tick_right()
                ax.set_yticks([0, 0.5, 1])
                ax.set_yticklabels(['0', '', '1'])#, ha='left', va='center')  # Align right
                for tick_label, tick_pos in zip(ax.get_yticklabels(), ax.get_yticks()):
                    if tick_pos == 0:
                        tick_label.set_va('bottom')
                    elif tick_pos == 0.5:
                        tick_label.set_va('center')
                    elif tick_pos == 1:
                        tick_label.set_va('top')
            else:
                ax.set_yticks([])

    # Set titles for each column of violin plots in the top row
    for i, l in enumerate(layout_abr.keys()):  # Flipped indexing
        axes[0, i].set_title(layout_abr[l]).set_fontsize(10)

    # Remove empty subplots
    for i in range(len(labels), len(axes)):  # Flipped indexing
        for j in range(len(dfs)):
            fig.delaxes(axes[i, j])

    # Adjust spacing
    fig.subplots_adjust(hspace=0.1)
    fig.subplots_adjust(wspace=0.1)

    #plt.savefig("Violin.pdf", format="pdf")
    plt.tight_layout()
    plt.show()

def violin_one_fig_transpose_nn():
    filename = "..\\Data\\Graph_Drawing_Metrics_All_with_NP.csv"
    df = pd.read_csv(filename)
    df = df.set_index("filename")

    # Create dataframes for each layout
    fr_df = df[df["layout"] == "Fruchterman-Reingold"]
    kk_df = df[df["layout"] == "Kamada-Kawai"]
    sugi_df = df[df["layout"] == "Sugiyama"]
    hola_df = df[df["layout"] == "HOLA"]
    dr_df = df[df["layout"] == "DRGraph"]
    ran_df = df[df["layout"] == "Random"]
    all_df = df[df["layout"] != "Random"]

    # print(min(hola_df["node_resolution"]))
    # print(max(hola_df["node_resolution"]))

    # Define the layouts and their corresponding dataframes
    dfs = {
        "Fruchterman-Reingold": fr_df,
        "Kamada-Kawai": kk_df,
        "Sugiyama": sugi_df,
        "HOLA": hola_df,
        "DRGraph": dr_df,
        "Random": ran_df,
        "All": all_df
    }

    layout_abr = {
        "Fruchterman-Reingold": "FR",
        "Kamada-Kawai": "KK",
        "Sugiyama": "Sugi",
        "HOLA": "HOLA",
        "DRGraph": "DRG",
        "Random": "Ran",
        "All": "All (excl. Ran)"
    }

    # Define the labels
    labels = ['angular_resolution', 'aspect_ratio', 'crossing_angle', 'edge_crossings', 'edge_length',
            'edge_orthogonality', 'gabriel_ratio', 'neighbourhood_preservation', 'node_resolution',
            'node_uniformity', 'stress']
    
    labels = ['stress', 'stress_not_normal']

    # Define the proper labels and abbreviations
    labels_proper = {
        'angular_resolution': 'Angular Resolution',
        'aspect_ratio': 'Aspect Ratio',
        'crossing_angle': 'Crossing Angle',
        'edge_crossings': 'Edge Crossings',
        'edge_length': 'Edge Lengths',
        'edge_orthogonality': 'Edge Orthogonality',
        'gabriel_ratio': 'Gabriel Ratio',
        'neighbourhood_preservation': 'Neighbourhood Preservation',
        'node_resolution': 'Node Resolution',
        'node_uniformity': 'Node Unifortmity',
        'stress': 'Stress'
    }

    labels_abr = {
        'angular_resolution': 'AR',
        'aspect_ratio': 'Asp',
        'crossing_angle': 'CA',
        'edge_crossings': 'EC',
        'edge_length': 'EL',
        'edge_orthogonality': 'EO',
        'gabriel_ratio': 'GR',
        'neighbourhood_preservation': 'NP',
        'node_resolution': 'NR',
        'node_uniformity': 'NU',
        'stress': 'Str'
    }

    labels_abr = {
        'stress': 'Str',
        'stress_not_normal':'StrNN'
    }
    from matplotlib.ticker import FixedLocator

    fig, axes = plt.subplots(len(labels), len(dfs), figsize=(16.5, 11.7))

    # Loop over the layouts
    for i, (layout, v) in enumerate(dfs.items()):
        v = v.drop(columns=['generator', 'layout', 'n', 'm', 'num_crossings'])
        #print(layout)

        # Retrieve data for each label
        for j, label in enumerate(labels):
            #print(label)
            data = v[label].dropna().to_numpy()

            # Create a violin plot for the current layout and label
            ax = axes[j, i]  # Flipped indexing
            #ax.violinplot(data, showmedians=True)  # Horizontal plot

            vp = ax.violinplot(data, showmedians=True)  # Horizontal plot

            # Adjust properties of the lines
            vp['cmedians'].set_linewidth(0.6)  # Fainter median line
            vp['cmins'].set_linewidth(0.6)  # Fainter minimum line
            vp['cmaxes'].set_linewidth(0.6)  # Fainter maximum line
            vp['cbars'].set_linewidth(0.6)  # Fainter quartile lines

            #ax.set_ylim(0, 1)
            # Remove the titles on each axis
            ax.set_title('')

            # Rotate the x-axis labels horizontally
            ax.tick_params(axis='y', rotation=0)
            ax.tick_params(axis='x', length=0)  # Remove vertical tick marks

            # Show tick marks only for the outer right and bottom plots
            #ax.set_yticklabels([], ha='left')  # Align left
            ax.set_xticklabels([], ha='center')  # Align center

            # Set y-axis label as layout name
            if i == 0:
                #ax.set_yticks([])
                ax.text(-0.025, 0.5, labels_abr[label], va='center', ha='right', fontsize=10, transform=ax.transAxes)

            # ax.set_xticks([])
            # Adjust x-axis tick alignment
            if i == len(dfs) - 1:
                ax.yaxis.tick_right()
                #ax.set_yticks([])
                # ax.set_yticks([0, 0.5, 1])
                # ax.set_yticklabels(['0', '', '1'])#, ha='left', va='center')  # Align right
                # for tick_label, tick_pos in zip(ax.get_yticklabels(), ax.get_yticks()):
                #     if tick_pos == 0:
                #         tick_label.set_va('bottom')
                #     elif tick_pos == 0.5:
                #         tick_label.set_va('center')
                #     elif tick_pos == 1:
                #         tick_label.set_va('top')
            else:
                ax.set_yticks([])

    # Set titles for each column of violin plots in the top row
    for i, l in enumerate(layout_abr.keys()):  # Flipped indexing
        axes[0, i].set_title(layout_abr[l]).set_fontsize(10)

    # Remove empty subplots
    for i in range(len(labels), len(axes)):  # Flipped indexing
        for j in range(len(dfs)):
            fig.delaxes(axes[i, j])

    # Adjust spacing
    fig.subplots_adjust(hspace=0.1)
    fig.subplots_adjust(wspace=0.1)

    #plt.savefig("Violin.pdf", format="pdf")
    plt.tight_layout()
    plt.show()

def violin_one_fig_transpose(filename):
    df = pd.read_csv(filename)
    df = df.set_index("filename")

    # Create dataframes for each layout
    fr_df = df[df["layout"] == "Fruchterman-Reingold"]
    kk_df = df[df["layout"] == "Kamada-Kawai"]
    dr_df = df[df["layout"] == "DRGraph"]
    sugi_df = df[df["layout"] == "Sugiyama"]
    hola_df = df[df["layout"] == "HOLA"]
    ran_df = df[df["layout"] == "Random"]
    all_df = df[df["layout"] != "Random"]

    # print(min(hola_df["node_resolution"]))
    # print(max(hola_df["node_resolution"]))

    # Define the layouts and their corresponding dataframes
    dfs = {
        "Fruchterman-Reingold": fr_df,
        "Kamada-Kawai": kk_df,
        "DRGraph": dr_df,
        "Sugiyama": sugi_df,
        "HOLA": hola_df,
        "Random": ran_df,
        "All": all_df
    }

    layout_abr = {
        "Fruchterman-Reingold": "FR",
        "Kamada-Kawai": "KK",
        "DRGraph": "DRG",
        "Sugiyama": "Sugi",
        "HOLA": "HOLA",
        "Random": "Ran",
        "All": "All (excl. Ran)"
    }

    # Define the labels
    labels = ['angular_resolution', 'aspect_ratio', 'crossing_angle', 'edge_crossings', 'centred_edge_crossings', 'edge_length', 'edge_orthogonality', 'gabriel_ratio', 
              'neighbourhood_preservation', 'node_resolution', 'node_uniformity', 'stress']

    # labels_proper = {"ar":"Angular Resolution", "asp":"Aspect Ratio", "ca":"Crossing Angle", "ec":"Edge Crossings", 
    #                  "el":"Edge Lengths", "eo":"Edge Orthogonality", "gr":"Gabriel Ratio", "np":"Neighbourhood Preservation", 
    #                  "nr":"Node Resolution", "nu":"Node Unifortmity"}

    labels_proper = {'angular_resolution':'Angular Resolution', 'aspect_ratio':'Aspect Ratio', 'crossing_angle':'Crossing Angle', 
                    'edge_crossings':'Edge Crossings', 'centred_edge_crossings': 'Edge Crossings (C)', 'edge_length':'Edge Lengths', 'edge_orthogonality':'Edge Orthogonality', 
                    'gabriel_ratio':'Gabriel Ratio', 'neighbourhood_preservation':'Neighbourhood Preservation', 
                    'node_resolution':'Node Resolution', 'node_uniformity':'Node Unifortmity', 'stress':'Stress'}
    
    labels_abr = {'angular_resolution':'AR', 'aspect_ratio':'Asp', 'crossing_angle':'CA', 
                    'edge_crossings':'EC', 'centred_edge_crossings': 'EC(C)', 'edge_length':'EL', 'edge_orthogonality':'EO', 
                    'gabriel_ratio':'GR', 'neighbourhood_preservation':'NP', 
                    'node_resolution':'NR', 'node_uniformity':'NU', 'stress':'Str'}


    from matplotlib.ticker import FixedLocator

    fig, axes = plt.subplots(len(labels), len(dfs), figsize=(16.5, 11.7))

    # Loop over the layouts
    for i, (layout, v) in enumerate(dfs.items()):
        v = v.drop(columns=['generator', 'layout', 'n', 'm', 'num_crossings'])
        #print(layout)

        # Retrieve data for each label
        for j, label in enumerate(labels):
            #print(label)
            data = v[label].dropna().to_numpy()

            # Create a violin plot for the current layout and label
            ax = axes[j, i]  # Flipped indexing
            #ax.violinplot(data, showmedians=True)  # Horizontal plot

            vp = ax.violinplot(data, showmedians=True)  # Horizontal plot

            # Adjust properties of the lines
            vp['cmedians'].set_linewidth(0.6)  # Fainter median line
            vp['cmins'].set_linewidth(0.6)  # Fainter minimum line
            vp['cmaxes'].set_linewidth(0.6)  # Fainter maximum line
            vp['cbars'].set_linewidth(0.6)  # Fainter quartile lines

            ax.set_ylim(0, 1)
            # Remove the titles on each axis
            ax.set_title('')

            # Rotate the x-axis labels horizontally
            ax.tick_params(axis='y', rotation=0)
            ax.tick_params(axis='x', length=0)  # Remove vertical tick marks

            # Show tick marks only for the outer right and bottom plots
            ax.set_yticklabels([], ha='left')  # Align left
            ax.set_xticklabels([], ha='center')  # Align center

            # Set y-axis label as layout name
            if i == 0:
                ax.set_yticks([])
                ax.text(-0.025, 0.5, labels_abr[label], va='center', ha='right', fontsize=10, transform=ax.transAxes)

            ax.set_xticks([])
            # Adjust x-axis tick alignment
            if i == len(dfs) - 1:
                ax.yaxis.tick_right()
                ax.set_yticks([0, 0.5, 1])
                ax.set_yticklabels(['0', '', '1'])#, ha='left', va='center')  # Align right
                for tick_label, tick_pos in zip(ax.get_yticklabels(), ax.get_yticks()):
                    if tick_pos == 0:
                        tick_label.set_va('bottom')
                    elif tick_pos == 0.5:
                        tick_label.set_va('center')
                    elif tick_pos == 1:
                        tick_label.set_va('top')
            else:
                ax.set_yticks([])

    # Set titles for each column of violin plots in the top row
    for i, l in enumerate(layout_abr.keys()):  # Flipped indexing
        axes[0, i].set_title(layout_abr[l]).set_fontsize(10)

    # Remove empty subplots
    for i in range(len(labels), len(axes)):  # Flipped indexing
        for j in range(len(dfs)):
            fig.delaxes(axes[i, j])

    # Adjust spacing
    fig.subplots_adjust(hspace=0.1)
    fig.subplots_adjust(wspace=0.1)

    plt.savefig("Violin_old.pdf", format="pdf")
    plt.tight_layout()
    plt.show()

def violin_one_fig_transpose_split(filename):

    # Generate example data
    
    df = pd.read_csv(filename)
    df = df.set_index("filename")

    df['nr_gen'] = df['generator'].map(lambda x: 'nr' if x in ['North', 'Rome'] else 'gen')

    df['blank'] = ""

    # rome_north_df = df[(df["generator"] == "Rome") | (df["generator"] == "North")]

    # not_rome_north_df = df[~((df["generator"] == "Rome") | (df["generator"] == "North"))]



    # # Create dataframes for each layout
    # fr_df = not_rome_north_df[not_rome_north_df["layout"] == "Fruchterman-Reingold"]
    # kk_df = not_rome_north_df[not_rome_north_df["layout"] == "Kamada-Kawai"]
    # dr_df = not_rome_north_df[not_rome_north_df["layout"] == "DRGraph"]
    # sugi_df = not_rome_north_df[not_rome_north_df["layout"] == "Sugiyama"]
    # hola_df = not_rome_north_df[not_rome_north_df["layout"] == "HOLA"]
    # ran_df = not_rome_north_df[not_rome_north_df["layout"] == "Random"]
    # all_df = not_rome_north_df[not_rome_north_df["layout"] != "Random"]

    # rn_fr_df = rome_north_df[rome_north_df["layout"] == "Fruchterman-Reingold"]
    # rn_kk_df = rome_north_df[rome_north_df["layout"] == "Kamada-Kawai"]
    # rn_dr_df = rome_north_df[rome_north_df["layout"] == "DRGraph"]
    # rn_sugi_df = rome_north_df[rome_north_df["layout"] == "Sugiyama"]
    # rn_hola_df = rome_north_df[rome_north_df["layout"] == "HOLA"]
    # rn_ran_df = rome_north_df[rome_north_df["layout"] == "Random"]
    # rn_all_df = rome_north_df[rome_north_df["layout"] != "Random"]

    fr_df = df[df["layout"] == "Fruchterman-Reingold"]
    kk_df = df[df["layout"] == "Kamada-Kawai"]
    dr_df = df[df["layout"] == "DRGraph"]
    sugi_df = df[df["layout"] == "Sugiyama"]
    hola_df = df[df["layout"] == "HOLA"]
    ran_df = df[df["layout"] == "Random"]
    all_df = df[df["layout"] != "Random"]


    # print(min(hola_df["node_resolution"]))
    # print(max(hola_df["node_resolution"]))

    # Define the layouts and their corresponding dataframes
    # dfs = {
    #     "Fruchterman-Reingold": [fr_df,  rn_fr_df],
    #     "Kamada-Kawai": [kk_df, rn_kk_df],
    #     "DRGraph": [dr_df, rn_dr_df],
    #     "Sugiyama": [sugi_df, rn_sugi_df],
    #     "HOLA": [hola_df, rn_hola_df],
    #     "Random": [ran_df, rn_ran_df],
    #     "All": [all_df, rn_all_df]
    # }
    dfs = {
        "Fruchterman-Reingold": fr_df,
        "Kamada-Kawai": kk_df,
        "DRGraph": dr_df,
        "Sugiyama": sugi_df,
        "HOLA": hola_df,
        "Random": ran_df,
        "All": all_df
    }

    layout_abr = {
        "Fruchterman-Reingold": "FR",
        "Kamada-Kawai": "KK",
        "DRGraph": "DRG",
        "Sugiyama": "Sugi",
        "HOLA": "HOLA",
        "Random": "Ran",
        "All": "All (excl. Ran)"
    }

    # Define the labels
    labels = ['angular_resolution', 'aspect_ratio', 'crossing_angle', 'edge_crossings', 'edge_length',
            'edge_orthogonality', 'gabriel_ratio', 
            #'neighbourhood_preservation',
            'node_resolution',
            'node_uniformity']
    

    # Define the proper labels and abbreviations
    labels_proper = {
        'angular_resolution': 'Angular Resolution',
        'aspect_ratio': 'Aspect Ratio',
        'crossing_angle': 'Crossing Angle',
        'edge_crossings': 'Edge Crossings',
        'edge_length': 'Edge Lengths',
        'edge_orthogonality': 'Edge Orthogonality',
        'gabriel_ratio': 'Gabriel Ratio',
        'neighbourhood_preservation': 'Neighbourhood Preservation',
        'node_resolution': 'Node Resolution',
        'node_uniformity': 'Node Unifortmity',
    }

    labels_abr = {
        'angular_resolution': 'AR',
        'aspect_ratio': 'Asp',
        'crossing_angle': 'CA',
        'edge_crossings': 'EC',
        'edge_length': 'EL',
        'edge_orthogonality': 'EO',
        'gabriel_ratio': 'GR',
        #'neighbourhood_preservation': 'NP',
        'node_resolution': 'NR',
        'node_uniformity': 'NU',
    }


    from matplotlib.ticker import FixedLocator

    fig, axes = plt.subplots(len(labels), len(dfs), figsize=(16.5, 11.7))

    # Loop over the layouts
    for i, (layout, v) in enumerate(dfs.items()):
        v = v.drop(columns=['generator', 'layout', 'n', 'm', 'num_crossings'])


        # Retrieve data for each label
        for j, label in enumerate(labels):
            data = v[label].dropna()

            # colors = ["skyblue", "salmon"]
            # colors = ["skyblue", "#FFA07A"]
            colors = ["tab:blue", "tab:red"]
            # Create a violin plot for the current layout and label
            ax = sns.violinplot(data=v, ax=axes[j, i], x="blank",  y=label, hue="nr_gen", saturation=0.5, split=True, inner="quartile", linewidth=0.1, palette=colors)
            # sns.pointplot(x = 'blank', y=label, data=df, estimator=np.median)
            for l in ax.lines:
                l.set_linestyle('--')
                l.set_linewidth(0.6)
                l.set_color('red')
                l.set_alpha(0.8)
            for l in ax.lines[1::3]:
                l.set_linestyle('-')
                l.set_linewidth(1.2)
                l.set_color('black')
                l.set_alpha(0.8)

            ax.set_xlabel("")
            ax.set_ylabel("")
            # ax.legend([],[], frameon=False)
            ax.legend_.remove()
            # Adjust properties of the lines
            # ax.lines[2].set_linewidth(0.6)  # Fainter median line
            # ax.lines[0].set_linewidth(0.6)  # Fainter minimum line
            # ax.lines[1].set_linewidth(0.6)  # Fainter maximum line

            # ax.set_ylim(-0.1, 1.1)
            ax.set_ylim(0, 1)
            ax.set_title('')

            # Rotate the x-axis labels horizontally
            ax.tick_params(axis='y', rotation=0)
            ax.tick_params(axis='x', length=0)  # Remove vertical tick marks

            # Show tick marks only for the outer right and bottom plots
            ax.set_yticklabels([], ha='left')  # Align left
            ax.set_xticklabels([], ha='center')  # Align center

            # Set y-axis label as layout name
            if i == 0:
                ax.set_yticks([])
                ax.text(-0.025, 0.5, labels_abr[label], va='center', ha='right', fontsize=10, transform=ax.transAxes)

            ax.set_xticks([])
            # Adjust x-axis tick alignment
            if i == len(dfs) - 1:
                ax.yaxis.tick_right()
                ax.set_yticks([0, 0.5, 1])
                ax.set_yticklabels(['0', '', '1'])
                for tick_label, tick_pos in zip(ax.get_yticklabels(), ax.get_yticks()):
                    if tick_pos == 0:
                        tick_label.set_va('bottom')
                    elif tick_pos == 0.5:
                        tick_label.set_va('center')
                    elif tick_pos == 1:
                        tick_label.set_va('top')
            else:
                ax.set_yticks([])

    # Set titles for each column of violin plots in the top row
    for i, l in enumerate(layout_abr.keys()):
        axes[0, i].set_title(layout_abr[l]).set_fontsize(10)

    # Remove empty subplots
    for i in range(len(labels), len(axes)):
        for j in range(len(dfs)):
            fig.delaxes(axes[i, j])

    # Adjust spacing
    fig.subplots_adjust(hspace=0.1)
    fig.subplots_adjust(wspace=0.1)

    plt.tight_layout()

    #plt.savefig("viol2.pdf", format="pdf")
    plt.show()

def violin_one_fig_transpose_split_none(filename):

    # Generate example data
    
    df = pd.read_csv(filename)
    df = df.set_index("filename")

    df['nr_gen'] = df['generator'].map(lambda x: 'nr' if x in ['North', 'Rome'] else 'gen')

    df['blank'] = ""

    df.drop(['stress', 'centred_edge_crossings'], axis=1)

    # df = df[~df.iloc[:, 1:].duplicated(keep='first')]
    df_no_hola_circ = df[~df['layout'].isin(["HOLA", "Circular"])]
    # print(df_no_hola_circ.head())
    df_hola_circ = df[df['layout'].isin(["HOLA", "Circular"])]
    #print(len(df_hola_circ))

    df_no_duplicates = df_no_hola_circ[~df_no_hola_circ.iloc[:, 1:].duplicated(keep='first')]

    duplicate_rows = df_no_hola_circ[df_no_hola_circ.iloc[:, 1:].duplicated(keep='first')]

    # Get the count of duplicate rows
    num_duplicate_rows = len(duplicate_rows)

    #print(f"Num duplicates: {num_duplicate_rows}")

    df = pd.concat([df_hola_circ, df_no_duplicates])

    # rome_north_df = df[(df["generator"] == "Rome") | (df["generator"] == "North")]

    # not_rome_north_df = df[~((df["generator"] == "Rome") | (df["generator"] == "North"))]



    # # Create dataframes for each layout
    # fr_df = not_rome_north_df[not_rome_north_df["layout"] == "Fruchterman-Reingold"]
    # kk_df = not_rome_north_df[not_rome_north_df["layout"] == "Kamada-Kawai"]
    # dr_df = not_rome_north_df[not_rome_north_df["layout"] == "DRGraph"]
    # sugi_df = not_rome_north_df[not_rome_north_df["layout"] == "Sugiyama"]
    # hola_df = not_rome_north_df[not_rome_north_df["layout"] == "HOLA"]
    # ran_df = not_rome_north_df[not_rome_north_df["layout"] == "Random"]
    # all_df = not_rome_north_df[not_rome_north_df["layout"] != "Random"]

    # rn_fr_df = rome_north_df[rome_north_df["layout"] == "Fruchterman-Reingold"]
    # rn_kk_df = rome_north_df[rome_north_df["layout"] == "Kamada-Kawai"]
    # rn_dr_df = rome_north_df[rome_north_df["layout"] == "DRGraph"]
    # rn_sugi_df = rome_north_df[rome_north_df["layout"] == "Sugiyama"]
    # rn_hola_df = rome_north_df[rome_north_df["layout"] == "HOLA"]
    # rn_ran_df = rome_north_df[rome_north_df["layout"] == "Random"]
    # rn_all_df = rome_north_df[rome_north_df["layout"] != "Random"]

    fr_df = df[df["layout"] == "Fruchterman-Reingold"]
    kk_df = df[df["layout"] == "Kamada-Kawai"]
    dr_df = df[df["layout"] == "DRGraph"]
    sugi_df = df[df["layout"] == "Sugiyama"]
    hola_df = df[df["layout"] == "HOLA"]
    circ_df = df[df["layout"] == "Circular"]
    ran_df = df[df["layout"] == "Random"]
    all_df = df[df["layout"] != "Random"]


    # print(min(hola_df["node_resolution"]))
    # print(max(hola_df["node_resolution"]))

    # Define the layouts and their corresponding dataframes
    # dfs = {
    #     "Fruchterman-Reingold": [fr_df,  rn_fr_df],
    #     "Kamada-Kawai": [kk_df, rn_kk_df],
    #     "DRGraph": [dr_df, rn_dr_df],
    #     "Sugiyama": [sugi_df, rn_sugi_df],
    #     "HOLA": [hola_df, rn_hola_df],
    #     "Random": [ran_df, rn_ran_df],
    #     "All": [all_df, rn_all_df]
    # }
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

    dfs1 = {
        "Fruchterman-Reingold": fr_df,
        "Kamada-Kawai": kk_df,
        "DRGraph": dr_df,
        "Sugiyama": sugi_df,
    }

    dfs2 = {
        "HOLA": hola_df,
        "Circular": circ_df,
        "Random": ran_df,
        "All": all_df
    }


    layout_abr = {
        "Fruchterman-Reingold": "FR",
        "Kamada-Kawai": "KK",
        "DRGraph": "DRGraph",
        "Sugiyama": "Sugi.",
        "HOLA": "HOLA",
        "Ciruclar": "Circ",
        "Random": "Ran.",
        "All": "All (excl. Ran.)"
    }

    layout_abr1 = {
        "Fruchterman-Reingold": "FR\n(83,840)",
        "Kamada-Kawai": "KK\n(80,721)",
        "DRGraph": "DRGraph\n(83,840)",
        "Sugiyama": "Sugiyama\n(82,157)",
    }

    layout_abr2 = {
        "HOLA": "HOLA\n(16,768)",
        "Ciruclar": "Circular\n(16,768)",
        "Random": "Random\n(83,840)",
        "All": "All (excl. Random)\n(364,094)"
    }

    # Define the labels
    labels = ['angular_resolution', 'aspect_ratio', 'crossing_angle', 'edge_crossings', 'edge_length',
            'edge_orthogonality', 'gabriel_ratio', 
            'neighbourhood_preservation',
            'node_resolution',
            'node_uniformity']
    

    # Define the proper labels and abbreviations
    labels_proper = {
        'angular_resolution': 'Angular Resolution',
        'aspect_ratio': 'Aspect Ratio',
        'crossing_angle': 'Crossing Angle',
        'edge_crossings': 'Edge Crossings',

        'edge_length': 'Edge Lengths',
        'edge_orthogonality': 'Edge Orthogonality',
        'gabriel_ratio': 'Gabriel Ratio',
        'neighbourhood_preservation': 'Neighbourhood Preservation',
        'node_resolution': 'Node Resolution',
        'node_uniformity': 'Node Unifortmity',

    }

    labels_abr = {
        'angular_resolution': 'AR',
        'aspect_ratio': 'Asp.',
        'crossing_angle': 'CA',
        'edge_crossings': 'EC',

        'edge_length': 'EL',
        'edge_orthogonality': 'EO',
        'gabriel_ratio': 'GR',
        'neighbourhood_preservation': 'NP',
        'node_resolution': 'NR',
        'node_uniformity': 'NU',

    }


    from matplotlib.ticker import FixedLocator

    fig, axes = plt.subplots(len(labels), len(dfs1), figsize=(9, 12))

    # Loop over the layouts
    for i, (layout, v) in enumerate(dfs1.items()):
        v = v.drop(columns=['generator', 'layout', 'n', 'm', 'num_crossings'])


        # Retrieve data for each label
        for j, label in enumerate(labels):
            data = v[label].dropna()

            colors = ["tab:blue"]#, "orange"]
            #colors = ["#4C72B0"]
            # Create a violin plot for the current layout and label
            ax = sns.violinplot(data=v, ax=axes[j, i], x="blank",  y=label, saturation=0.5, inner="quartile", linewidth=0.1, palette=colors)
            # sns.pointplot(x = 'blank', y=label, data=df, estimator=np.median)
            for l in ax.lines:
                l.set_linestyle('--')
                l.set_linewidth(0.6)
                l.set_color('red')
                l.set_alpha(0.8)
            for l in ax.lines[1::3]:
                l.set_linestyle('-')
                l.set_linewidth(1.2)
                l.set_color('black')
                l.set_alpha(0.8)

            ax.set_xlabel("")
            ax.set_ylabel("")
            # ax.legend([],[], frameon=False)
            #ax.legend_.remove()
            # Adjust properties of the lines
            # ax.lines[2].set_linewidth(0.6)  # Fainter median line
            # ax.lines[0].set_linewidth(0.6)  # Fainter minimum line
            # ax.lines[1].set_linewidth(0.6)  # Fainter maximum line

            ax.set_ylim(-0.1, 1.1)
            #ax.set_ylim(0, 1)
            ax.set_title('')

            # Rotate the x-axis labels horizontally
            ax.tick_params(axis='y', rotation=0)
            ax.tick_params(axis='x', length=0)  # Remove vertical tick marks

            # Show tick marks only for the outer right and bottom plots
            ax.set_yticklabels([], ha='left')  # Align left
            ax.set_xticklabels([], ha='center')  # Align center

            # Set y-axis label as layout name
            if i == 0:
                ax.set_yticks([])
                ax.text(-0.025, 0.5, labels_abr[label], va='center', ha='right', fontsize=10, transform=ax.transAxes)

            ax.set_xticks([])
            # Adjust x-axis tick alignment
            if i == len(dfs1) - 1:
                ax.yaxis.tick_right()
                ax.set_yticks([0, 0.5, 1])
                ax.set_yticklabels(['0', '', '1'])
                for tick_label, tick_pos in zip(ax.get_yticklabels(), ax.get_yticks()):
                    if tick_pos == 0:
                        tick_label.set_va('bottom')
                    elif tick_pos == 0.5:
                        tick_label.set_va('center')
                    elif tick_pos == 1:
                        tick_label.set_va('top')
            else:
                ax.set_yticks([])

    # Set titles for each column of violin plots in the top row
    for i, l in enumerate(layout_abr1.keys()):
        axes[0, i].set_title(layout_abr1[l]).set_fontsize(10)

    # Remove empty subplots
    for i in range(len(labels), len(axes)):
        for j in range(len(dfs1)):
            fig.delaxes(axes[i, j])

    # Adjust spacing
    fig.subplots_adjust(hspace=0.1)
    fig.subplots_adjust(wspace=0.1)

    plt.tight_layout()

    plt.savefig("violin_all_1.pdf", format="pdf")
    plt.show()


    #############################################################################################

    fig, axes = plt.subplots(len(labels), len(dfs2), figsize=(9, 12))

    # Loop over the layouts
    for i, (layout, v) in enumerate(dfs2.items()):
        v = v.drop(columns=['generator', 'layout', 'n', 'm', 'num_crossings'])


        # Retrieve data for each label
        for j, label in enumerate(labels):
            data = v[label].dropna()

            colors = ["tab:blue"]#, "orange"]
            #colors = ["#4C72B0"]
            # Create a violin plot for the current layout and label
            ax = sns.violinplot(data=v, ax=axes[j, i], x="blank",  y=label, saturation=0.5, inner="quartile", linewidth=0.1, palette=colors)
            # sns.pointplot(x = 'blank', y=label, data=df, estimator=np.median)
            for l in ax.lines:
                l.set_linestyle('--')
                l.set_linewidth(0.6)
                l.set_color('red')
                l.set_alpha(0.8)
            for l in ax.lines[1::3]:
                l.set_linestyle('-')
                l.set_linewidth(1.2)
                l.set_color('black')
                l.set_alpha(0.8)

            ax.set_xlabel("")
            ax.set_ylabel("")
            # ax.legend([],[], frameon=False)
            #ax.legend_.remove()
            # Adjust properties of the lines
            # ax.lines[2].set_linewidth(0.6)  # Fainter median line
            # ax.lines[0].set_linewidth(0.6)  # Fainter minimum line
            # ax.lines[1].set_linewidth(0.6)  # Fainter maximum line

            ax.set_ylim(-0.1, 1.1)
            #ax.set_ylim(0, 1)
            ax.set_title('')

            # Rotate the x-axis labels horizontally
            ax.tick_params(axis='y', rotation=0)
            ax.tick_params(axis='x', length=0)  # Remove vertical tick marks

            # Show tick marks only for the outer right and bottom plots
            ax.set_yticklabels([], ha='left')  # Align left
            ax.set_xticklabels([], ha='center')  # Align center

            # Set y-axis label as layout name
            if i == 0:
                ax.set_yticks([])
                ax.text(-0.025, 0.5, labels_abr[label], va='center', ha='right', fontsize=10, transform=ax.transAxes)

            ax.set_xticks([])
            # Adjust x-axis tick alignment
            if i == len(dfs2) - 1:
                ax.yaxis.tick_right()
                ax.set_yticks([0, 0.5, 1])
                ax.set_yticklabels(['0', '', '1'])
                for tick_label, tick_pos in zip(ax.get_yticklabels(), ax.get_yticks()):
                    if tick_pos == 0:
                        tick_label.set_va('bottom')
                    elif tick_pos == 0.5:
                        tick_label.set_va('center')
                    elif tick_pos == 1:
                        tick_label.set_va('top')
            else:
                ax.set_yticks([])

    # Set titles for each column of violin plots in the top row
    for i, l in enumerate(layout_abr2.keys()):
        axes[0, i].set_title(layout_abr2[l]).set_fontsize(10)

    # Remove empty subplots
    for i in range(len(labels), len(axes)):
        for j in range(len(dfs2)):
            fig.delaxes(axes[i, j])

    # Adjust spacing
    fig.subplots_adjust(hspace=0.1)
    fig.subplots_adjust(wspace=0.1)

    plt.tight_layout()

    plt.savefig("violin_all_2.pdf", format="pdf")
    plt.show()

def violin_one_fig_transpose_split_north(filename):

    # Generate example data
    
    df = pd.read_csv(filename)
    df = df.set_index("filename")
    df = df.drop(df[df["generator"] == "Rome"].index)

    df['nr_gen'] = df['generator'].map(lambda x: 'nr' if x in ['North'] else 'gen')

    df['blank'] = ""

    # rome_north_df = df[(df["generator"] == "Rome") | (df["generator"] == "North")]

    # not_rome_north_df = df[~((df["generator"] == "Rome") | (df["generator"] == "North"))]



    # # Create dataframes for each layout
    # fr_df = not_rome_north_df[not_rome_north_df["layout"] == "Fruchterman-Reingold"]
    # kk_df = not_rome_north_df[not_rome_north_df["layout"] == "Kamada-Kawai"]
    # dr_df = not_rome_north_df[not_rome_north_df["layout"] == "DRGraph"]
    # sugi_df = not_rome_north_df[not_rome_north_df["layout"] == "Sugiyama"]
    # hola_df = not_rome_north_df[not_rome_north_df["layout"] == "HOLA"]
    # ran_df = not_rome_north_df[not_rome_north_df["layout"] == "Random"]
    # all_df = not_rome_north_df[not_rome_north_df["layout"] != "Random"]

    # rn_fr_df = rome_north_df[rome_north_df["layout"] == "Fruchterman-Reingold"]
    # rn_kk_df = rome_north_df[rome_north_df["layout"] == "Kamada-Kawai"]
    # rn_dr_df = rome_north_df[rome_north_df["layout"] == "DRGraph"]
    # rn_sugi_df = rome_north_df[rome_north_df["layout"] == "Sugiyama"]
    # rn_hola_df = rome_north_df[rome_north_df["layout"] == "HOLA"]
    # rn_ran_df = rome_north_df[rome_north_df["layout"] == "Random"]
    # rn_all_df = rome_north_df[rome_north_df["layout"] != "Random"]

    fr_df = df[df["layout"] == "Fruchterman-Reingold"]
    kk_df = df[df["layout"] == "Kamada-Kawai"]
    dr_df = df[df["layout"] == "DRGraph"]
    sugi_df = df[df["layout"] == "Sugiyama"]
    hola_df = df[df["layout"] == "HOLA"]
    ran_df = df[df["layout"] == "Random"]
    all_df = df[df["layout"] != "Random"]


    # print(min(hola_df["node_resolution"]))
    # print(max(hola_df["node_resolution"]))

    # Define the layouts and their corresponding dataframes
    # dfs = {
    #     "Fruchterman-Reingold": [fr_df,  rn_fr_df],
    #     "Kamada-Kawai": [kk_df, rn_kk_df],
    #     "DRGraph": [dr_df, rn_dr_df],
    #     "Sugiyama": [sugi_df, rn_sugi_df],
    #     "HOLA": [hola_df, rn_hola_df],
    #     "Random": [ran_df, rn_ran_df],
    #     "All": [all_df, rn_all_df]
    # }
    dfs = {
        "Fruchterman-Reingold": fr_df,
        "Kamada-Kawai": kk_df,
        "DRGraph": dr_df,
        "Sugiyama": sugi_df,
        "HOLA": hola_df,
        "Random": ran_df,
        "All": all_df
    }

    layout_abr = {
        "Fruchterman-Reingold": "FR",
        "Kamada-Kawai": "KK",
        "DRGraph": "DRG",
        "Sugiyama": "Sugi",
        "HOLA": "HOLA",
        "Random": "Ran",
        "All": "All (excl. Ran)"
    }

    # Define the labels
    labels = ['angular_resolution', 'aspect_ratio', 'crossing_angle', 'edge_crossings', 'edge_length',
            'edge_orthogonality', 'gabriel_ratio', 
            'neighbourhood_preservation',
            'node_resolution',
            'node_uniformity']
    

    # Define the proper labels and abbreviations
    labels_proper = {
        'angular_resolution': 'Angular Resolution',
        'aspect_ratio': 'Aspect Ratio',
        'crossing_angle': 'Crossing Angle',
        'edge_crossings': 'Edge Crossings',
        'edge_length': 'Edge Lengths',
        'edge_orthogonality': 'Edge Orthogonality',
        'gabriel_ratio': 'Gabriel Ratio',
        'neighbourhood_preservation': 'Neighbourhood Preservation',
        'node_resolution': 'Node Resolution',
        'node_uniformity': 'Node Unifortmity',
    }

    labels_abr = {
        'angular_resolution': 'AR',
        'aspect_ratio': 'Asp',
        'crossing_angle': 'CA',
        'edge_crossings': 'EC',
        'edge_length': 'EL',
        'edge_orthogonality': 'EO',
        'gabriel_ratio': 'GR',
        'neighbourhood_preservation': 'NP',
        'node_resolution': 'NR',
        'node_uniformity': 'NU',
    }


    from matplotlib.ticker import FixedLocator

    fig, axes = plt.subplots(len(labels), len(dfs), figsize=(16.5, 11.7))

    # Loop over the layouts
    for i, (layout, v) in enumerate(dfs.items()):
        v = v.drop(columns=['generator', 'layout', 'n', 'm', 'num_crossings'])


        # Retrieve data for each label
        for j, label in enumerate(labels):
            data = v[label].dropna()

            # colors = ["skyblue", "salmon"]
            # colors = ["skyblue", "#FFA07A"]
            colors = ["tab:blue", "tab:orange"]
            # Create a violin plot for the current layout and label
            ax = sns.violinplot(data=v, ax=axes[j, i], x="blank",  y=label, hue="nr_gen", saturation=0.5, split=True, inner="quartile", linewidth=0.1, palette=colors)
            # sns.pointplot(x = 'blank', y=label, data=df, estimator=np.median)
            for l in ax.lines:
                l.set_linestyle('--')
                l.set_linewidth(0.6)
                l.set_color('red')
                l.set_alpha(0.8)
            for l in ax.lines[1::3]:
                l.set_linestyle('-')
                l.set_linewidth(1.2)
                l.set_color('black')
                l.set_alpha(0.8)

            ax.set_xlabel("")
            ax.set_ylabel("")
            # ax.legend([],[], frameon=False)
            ax.legend_.remove()
            # Adjust properties of the lines
            # ax.lines[2].set_linewidth(0.6)  # Fainter median line
            # ax.lines[0].set_linewidth(0.6)  # Fainter minimum line
            # ax.lines[1].set_linewidth(0.6)  # Fainter maximum line

            # ax.set_ylim(-0.1, 1.1)
            ax.set_ylim(0, 1)
            ax.set_title('')

            # Rotate the x-axis labels horizontally
            ax.tick_params(axis='y', rotation=0)
            ax.tick_params(axis='x', length=0)  # Remove vertical tick marks

            # Show tick marks only for the outer right and bottom plots
            ax.set_yticklabels([], ha='left')  # Align left
            ax.set_xticklabels([], ha='center')  # Align center

            # Set y-axis label as layout name
            if i == 0:
                ax.set_yticks([])
                ax.text(-0.025, 0.5, labels_abr[label], va='center', ha='right', fontsize=10, transform=ax.transAxes)

            ax.set_xticks([])
            # Adjust x-axis tick alignment
            if i == len(dfs) - 1:
                ax.yaxis.tick_right()
                ax.set_yticks([0, 0.5, 1])
                ax.set_yticklabels(['0', '', '1'])
                for tick_label, tick_pos in zip(ax.get_yticklabels(), ax.get_yticks()):
                    if tick_pos == 0:
                        tick_label.set_va('bottom')
                    elif tick_pos == 0.5:
                        tick_label.set_va('center')
                    elif tick_pos == 1:
                        tick_label.set_va('top')
            else:
                ax.set_yticks([])

    # Set titles for each column of violin plots in the top row
    for i, l in enumerate(layout_abr.keys()):
        axes[0, i].set_title(layout_abr[l]).set_fontsize(10)

    # Remove empty subplots
    for i in range(len(labels), len(axes)):
        for j in range(len(dfs)):
            fig.delaxes(axes[i, j])

    # Adjust spacing
    fig.subplots_adjust(hspace=0.1)
    fig.subplots_adjust(wspace=0.1)

    plt.tight_layout()

    plt.savefig("violin_north.pdf", format="pdf")
    plt.show()


def violin_one_fig_transpose_split_rome(filename):

    # Generate example data
    
    df = pd.read_csv(filename)
    df = df.set_index("filename")
    df = df.drop(df[df["generator"] == "North"].index)

    df['nr_gen'] = df['generator'].map(lambda x: 'nr' if x in ['Rome'] else 'gen')

    df['blank'] = ""

    # rome_north_df = df[(df["generator"] == "Rome") | (df["generator"] == "North")]

    # not_rome_north_df = df[~((df["generator"] == "Rome") | (df["generator"] == "North"))]



    # # Create dataframes for each layout
    # fr_df = not_rome_north_df[not_rome_north_df["layout"] == "Fruchterman-Reingold"]
    # kk_df = not_rome_north_df[not_rome_north_df["layout"] == "Kamada-Kawai"]
    # dr_df = not_rome_north_df[not_rome_north_df["layout"] == "DRGraph"]
    # sugi_df = not_rome_north_df[not_rome_north_df["layout"] == "Sugiyama"]
    # hola_df = not_rome_north_df[not_rome_north_df["layout"] == "HOLA"]
    # ran_df = not_rome_north_df[not_rome_north_df["layout"] == "Random"]
    # all_df = not_rome_north_df[not_rome_north_df["layout"] != "Random"]

    # rn_fr_df = rome_north_df[rome_north_df["layout"] == "Fruchterman-Reingold"]
    # rn_kk_df = rome_north_df[rome_north_df["layout"] == "Kamada-Kawai"]
    # rn_dr_df = rome_north_df[rome_north_df["layout"] == "DRGraph"]
    # rn_sugi_df = rome_north_df[rome_north_df["layout"] == "Sugiyama"]
    # rn_hola_df = rome_north_df[rome_north_df["layout"] == "HOLA"]
    # rn_ran_df = rome_north_df[rome_north_df["layout"] == "Random"]
    # rn_all_df = rome_north_df[rome_north_df["layout"] != "Random"]

    fr_df = df[df["layout"] == "Fruchterman-Reingold"]
    kk_df = df[df["layout"] == "Kamada-Kawai"]
    dr_df = df[df["layout"] == "DRGraph"]
    sugi_df = df[df["layout"] == "Sugiyama"]
    hola_df = df[df["layout"] == "HOLA"]
    ran_df = df[df["layout"] == "Random"]
    all_df = df[df["layout"] != "Random"]


    # print(min(hola_df["node_resolution"]))
    # print(max(hola_df["node_resolution"]))

    # Define the layouts and their corresponding dataframes
    # dfs = {
    #     "Fruchterman-Reingold": [fr_df,  rn_fr_df],
    #     "Kamada-Kawai": [kk_df, rn_kk_df],
    #     "DRGraph": [dr_df, rn_dr_df],
    #     "Sugiyama": [sugi_df, rn_sugi_df],
    #     "HOLA": [hola_df, rn_hola_df],
    #     "Random": [ran_df, rn_ran_df],
    #     "All": [all_df, rn_all_df]
    # }
    dfs = {
        "Fruchterman-Reingold": fr_df,
        "Kamada-Kawai": kk_df,
        "DRGraph": dr_df,
        "Sugiyama": sugi_df,
        "HOLA": hola_df,
        "Random": ran_df,
        "All": all_df
    }

    layout_abr = {
        "Fruchterman-Reingold": "FR",
        "Kamada-Kawai": "KK",
        "DRGraph": "DRG",
        "Sugiyama": "Sugi",
        "HOLA": "HOLA",
        "Random": "Ran",
        "All": "All (excl. Ran)"
    }

    # Define the labels
    labels = ['angular_resolution', 'aspect_ratio', 'crossing_angle', 'edge_crossings', 'edge_length',
            'edge_orthogonality', 'gabriel_ratio', 
            'neighbourhood_preservation',
            'node_resolution',
            'node_uniformity']
    

    # Define the proper labels and abbreviations
    labels_proper = {
        'angular_resolution': 'Angular Resolution',
        'aspect_ratio': 'Aspect Ratio',
        'crossing_angle': 'Crossing Angle',
        'edge_crossings': 'Edge Crossings',
        'edge_length': 'Edge Lengths',
        'edge_orthogonality': 'Edge Orthogonality',
        'gabriel_ratio': 'Gabriel Ratio',
        'neighbourhood_preservation': 'Neighbourhood Preservation',
        'node_resolution': 'Node Resolution',
        'node_uniformity': 'Node Unifortmity',
    }

    labels_abr = {
        'angular_resolution': 'AR',
        'aspect_ratio': 'Asp',
        'crossing_angle': 'CA',
        'edge_crossings': 'EC',
        'edge_length': 'EL',
        'edge_orthogonality': 'EO',
        'gabriel_ratio': 'GR',
        'neighbourhood_preservation': 'NP',
        'node_resolution': 'NR',
        'node_uniformity': 'NU',
    }


    from matplotlib.ticker import FixedLocator

    fig, axes = plt.subplots(len(labels), len(dfs), figsize=(16.5, 11.7))

    # Loop over the layouts
    for i, (layout, v) in enumerate(dfs.items()):
        v = v.drop(columns=['generator', 'layout', 'n', 'm', 'num_crossings'])


        # Retrieve data for each label
        for j, label in enumerate(labels):
            data = v[label].dropna()

            # colors = ["skyblue", "salmon"]
            # colors = ["skyblue", "#FFA07A"]
            colors = ["tab:blue", "tab:red"]
            # Create a violin plot for the current layout and label
            ax = sns.violinplot(data=v, ax=axes[j, i], x="blank",  y=label, hue="nr_gen", saturation=0.5, split=True, inner="quartile", linewidth=0.1, palette=colors)
            # sns.pointplot(x = 'blank', y=label, data=df, estimator=np.median)
            for l in ax.lines:
                l.set_linestyle('--')
                l.set_linewidth(0.6)
                l.set_color('red')
                l.set_alpha(0.8)
            for l in ax.lines[1::3]:
                l.set_linestyle('-')
                l.set_linewidth(1.2)
                l.set_color('black')
                l.set_alpha(0.8)

            ax.set_xlabel("")
            ax.set_ylabel("")
            # ax.legend([],[], frameon=False)
            ax.legend_.remove()
            # Adjust properties of the lines
            # ax.lines[2].set_linewidth(0.6)  # Fainter median line
            # ax.lines[0].set_linewidth(0.6)  # Fainter minimum line
            # ax.lines[1].set_linewidth(0.6)  # Fainter maximum line

            # ax.set_ylim(-0.1, 1.1)
            ax.set_ylim(0, 1)
            ax.set_title('')

            # Rotate the x-axis labels horizontally
            ax.tick_params(axis='y', rotation=0)
            ax.tick_params(axis='x', length=0)  # Remove vertical tick marks

            # Show tick marks only for the outer right and bottom plots
            ax.set_yticklabels([], ha='left')  # Align left
            ax.set_xticklabels([], ha='center')  # Align center

            # Set y-axis label as layout name
            if i == 0:
                ax.set_yticks([])
                ax.text(-0.025, 0.5, labels_abr[label], va='center', ha='right', fontsize=10, transform=ax.transAxes)

            ax.set_xticks([])
            # Adjust x-axis tick alignment
            if i == len(dfs) - 1:
                ax.yaxis.tick_right()
                ax.set_yticks([0, 0.5, 1])
                ax.set_yticklabels(['0', '', '1'])
                for tick_label, tick_pos in zip(ax.get_yticklabels(), ax.get_yticks()):
                    if tick_pos == 0:
                        tick_label.set_va('bottom')
                    elif tick_pos == 0.5:
                        tick_label.set_va('center')
                    elif tick_pos == 1:
                        tick_label.set_va('top')
            else:
                ax.set_yticks([])

    # Set titles for each column of violin plots in the top row
    for i, l in enumerate(layout_abr.keys()):
        axes[0, i].set_title(layout_abr[l]).set_fontsize(10)

    # Remove empty subplots
    for i in range(len(labels), len(axes)):
        for j in range(len(dfs)):
            fig.delaxes(axes[i, j])

    # Adjust spacing
    fig.subplots_adjust(hspace=0.1)
    fig.subplots_adjust(wspace=0.1)

    plt.tight_layout()

    plt.savefig("violin_rome.pdf", format="pdf")
    plt.show()

def violin_one_fig_transpose_split_rome_north(filename):

    # Generate example data
    
    df = pd.read_csv(filename)
    df = df.set_index("filename")
    #df = df.drop(df[df["generator"] == "North"].index)
    df = df.drop(df[df["layout"] == "Random"].index)

    # df = df[~df.iloc[:, 1:].duplicated(keep='first')]
    df_no_hola_circ = df[~df['layout'].isin(["HOLA", "Circular"])]
    # print(df_no_hola_circ.head())
    df_hola_circ = df[df['layout'].isin(["HOLA", "Circular"])]
    #print(len(df_hola_circ))

    df_no_duplicates = df_no_hola_circ[~df_no_hola_circ.iloc[:, 1:].duplicated(keep='first')]

    duplicate_rows = df_no_hola_circ[df_no_hola_circ.iloc[:, 1:].duplicated(keep='first')]

    # Get the count of duplicate rows
    num_duplicate_rows = len(duplicate_rows)

    #print(f"Num duplicates: {num_duplicate_rows}")

    df = pd.concat([df_hola_circ, df_no_duplicates])

    rome_df = df.copy()
    rome_df = rome_df.drop(df[df["generator"] == "North"].index)

    north_df = df.copy()
    north_df = north_df.drop(df[df["generator"] == "Rome"].index)

    north_df['nr_gen'] = north_df['generator'].map(lambda x: 'nr' if x in ['North'] else 'gen')
    rome_df['nr_gen'] = rome_df['generator'].map(lambda x: 'nr' if x in ['Rome'] else 'gen')

    north_df['blank'] = ""
    rome_df['blank'] = ""

    # rome_north_df = df[(df["generator"] == "Rome") | (df["generator"] == "North")]

    # not_rome_north_df = df[~((df["generator"] == "Rome") | (df["generator"] == "North"))]



    # # Create dataframes for each layout
    # fr_df = not_rome_north_df[not_rome_north_df["layout"] == "Fruchterman-Reingold"]
    # kk_df = not_rome_north_df[not_rome_north_df["layout"] == "Kamada-Kawai"]
    # dr_df = not_rome_north_df[not_rome_north_df["layout"] == "DRGraph"]
    # sugi_df = not_rome_north_df[not_rome_north_df["layout"] == "Sugiyama"]
    # hola_df = not_rome_north_df[not_rome_north_df["layout"] == "HOLA"]
    # ran_df = not_rome_north_df[not_rome_north_df["layout"] == "Random"]
    # all_df = not_rome_north_df[not_rome_north_df["layout"] != "Random"]

    # rn_fr_df = rome_north_df[rome_north_df["layout"] == "Fruchterman-Reingold"]
    # rn_kk_df = rome_north_df[rome_north_df["layout"] == "Kamada-Kawai"]
    # rn_dr_df = rome_north_df[rome_north_df["layout"] == "DRGraph"]
    # rn_sugi_df = rome_north_df[rome_north_df["layout"] == "Sugiyama"]
    # rn_hola_df = rome_north_df[rome_north_df["layout"] == "HOLA"]
    # rn_ran_df = rome_north_df[rome_north_df["layout"] == "Random"]
    # rn_all_df = rome_north_df[rome_north_df["layout"] != "Random"]

    # fr_df = df[df["layout"] == "Fruchterman-Reingold"]
    # kk_df = df[df["layout"] == "Kamada-Kawai"]
    # dr_df = df[df["layout"] == "DRGraph"]
    # sugi_df = df[df["layout"] == "Sugiyama"]
    # hola_df = df[df["layout"] == "HOLA"]
    # ran_df = df[df["layout"] == "Random"]
    #all_df = df[df["layout"] != "Random"]


    # print(min(hola_df["node_resolution"]))
    # print(max(hola_df["node_resolution"]))

    # Define the layouts and their corresponding dataframes
    # dfs = {
    #     "Fruchterman-Reingold": [fr_df,  rn_fr_df],
    #     "Kamada-Kawai": [kk_df, rn_kk_df],
    #     "DRGraph": [dr_df, rn_dr_df],
    #     "Sugiyama": [sugi_df, rn_sugi_df],
    #     "HOLA": [hola_df, rn_hola_df],
    #     "Random": [ran_df, rn_ran_df],
    #     "All": [all_df, rn_all_df]
    # }
    dfs = {
        # "Fruchterman-Reingold": fr_df,
        # "Kamada-Kawai": kk_df,
        # "DRGraph": dr_df,
        # "Sugiyama": sugi_df,
        # "HOLA": hola_df,
        # "Random": ran_df,
        # "All_rome": all_df
        "Generated/North": north_df,
        "Generated/Rome": rome_df
    }

    layout_abr = {
        # "Fruchterman-Reingold": "FR",
        # "Kamada-Kawai": "KK",
        # "DRGraph": "DRG",
        # "Sugiyama": "Sugi",
        # "HOLA": "HOLA",
        # "Random": "Ran",
        # "All": "All (excl. Ran)"
        "Generated/North": "North/Generated (excl. Ran)",
        "Generated/Rome": "Rome/Generated (excl. Ran)"
    }

    # Define the labels
    labels = ['angular_resolution', 'aspect_ratio', 'crossing_angle', 'edge_crossings', 'edge_length',
            'edge_orthogonality', 'gabriel_ratio', 
            'neighbourhood_preservation',
            'node_resolution',
            'node_uniformity']
    

    # Define the proper labels and abbreviations
    labels_proper = {
        'angular_resolution': 'Angular Resolution',
        'aspect_ratio': 'Aspect Ratio',
        'crossing_angle': 'Crossing Angle',
        'edge_crossings': 'Edge Crossings',
        'edge_length': 'Edge Lengths',
        'edge_orthogonality': 'Edge Orthogonality',
        'gabriel_ratio': 'Gabriel Ratio',
        'neighbourhood_preservation': 'Neighbourhood Preservation',
        'node_resolution': 'Node Resolution',
        'node_uniformity': 'Node Unifortmity',
    }

    labels_abr = {
        'angular_resolution': 'AR',
        'aspect_ratio': 'Asp',
        'crossing_angle': 'CA',
        'edge_crossings': 'EC',
        'edge_length': 'EL',
        'edge_orthogonality': 'EO',
        'gabriel_ratio': 'GR',
        'neighbourhood_preservation': 'NP',
        'node_resolution': 'NR',
        'node_uniformity': 'NU',
    }


    from matplotlib.ticker import FixedLocator

    fig, axes = plt.subplots(len(dfs), len(labels), figsize=(16.5, 6))

    # Loop over the layouts
    for i, (layout, v) in enumerate(dfs.items()):

        v = v.drop(columns=['generator', 'layout', 'n', 'm', 'num_crossings'])

        # Retrieve data for each label
        for j, label in enumerate(labels):
            data = v[label].dropna()

            if i == 0:
                colors = ["tab:blue", "tab:orange"]
            else:
                colors = ["tab:blue", "tab:red"]

            # Create a violin plot for the current layout and label
            ax = sns.violinplot(data=v, ax=axes[i, j], y="blank", x=label, hue="nr_gen", saturation=0.5, split=True, inner="quartile", linewidth=0.1, palette=colors)

            for l in ax.lines:
                l.set_linestyle('--')
                l.set_linewidth(0.6)
                l.set_color('red')
                l.set_alpha(0.8)
            for l in ax.lines[1::3]:
                l.set_linestyle('-')
                l.set_linewidth(1.2)
                l.set_color('black')
                l.set_alpha(0.8)

            ax.set_xlabel("")
            ax.set_ylabel("")
            ax.legend_.remove()
            ax.set_xlim(-0.1, 1.1)
            #ax.set_xlim(0, 1)
            ax.set_title('')

            # Rotate the x-axis labels vertically
            ax.tick_params(axis='x', rotation=0)
            ax.tick_params(axis='y', length=0)  # Remove horizontal tick marks

            # Show tick marks only for the outer top and right plots
            ax.set_xticklabels([], ha='center')  # Align center
            # if j == len(labels) - 1:
            # ax.xaxis.tick_top()
            ax.set_xticks([0, 0.5, 1])
            ax.set_xticklabels(['0', '', '1'])
            for tick_label, tick_pos in zip(ax.get_xticklabels(), ax.get_xticks()):
                if tick_pos == 0:
                    tick_label.set_ha('left')
                elif tick_pos == 0.5:
                    tick_label.set_ha('center')
                elif tick_pos == 1:
                    tick_label.set_ha('right')
            # else:
            #     ax.set_xticks([])

            # Set y-axis label as layout name
            if j == 0:
                #ax.set_xticks([])
                ax.set_ylabel(layout_abr[layout])
                #ax.text(0.5, -0.15, labels_abr[label], va='center', ha='center', fontsize=10, transform=ax.transAxes)

            ax.set_yticks([])

    # Set titles for each row of violin plots in the leftmost column
    for i, l in enumerate(labels_abr):
        axes[0, i].set_title(labels_abr[l]).set_fontsize(10)

    # Remove empty subplots
    for i in range(len(dfs), len(axes)):
        for j in range(len(labels)):
            fig.delaxes(axes[i, j])

    # Adjust spacing
    fig.subplots_adjust(hspace=0.1)
    fig.subplots_adjust(wspace=0.1)

    plt.tight_layout()
    plt.savefig("violin_north_rome.pdf", format="pdf")
    plt.show()

def numeric_distribution(filename):
    df = pd.read_csv(filename)
    df = df.set_index("filename")

    # Create dataframes for each layout
    fr_df = df[df["layout"] == "Fruchterman-Reingold"]
    kk_df = df[df["layout"] == "Kamada-Kawai"]
    sugi_df = df[df["layout"] == "Sugiyama"]
    hola_df = df[df["layout"] == "HOLA"]
    dr_df = df[df["layout"] == "DRGraph"]
    circ_df = df[df["layout"] == "Circular"]
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
        "DRGraph": "DRG",
        "Sugiyama": "Sugi",
        "HOLA": "HOLA",
        "Circular": "Circ",
        "Random": "Ran",
        "All": "All (excl. Ran)"
    }

    # Define the labels
    labels = ['angular_resolution', 'aspect_ratio', 'crossing_angle', 'edge_crossings', 'edge_length',
            'edge_orthogonality', 'gabriel_ratio', 'neighbourhood_preservation', 'node_resolution',
            'node_uniformity' ]


    values = [["" for _ in range(len(dfs) + 1)] for _ in range(len(labels)+1)]

    
    #print(values)
    for i, label in enumerate(labels):
        values[i + 1][0] = label  # Set the label in the first column

        for j, (layout, df) in enumerate(dfs.items()):
            values[0][j + 1] = layout  # Set the layout name in the first row
            quartiles = np.percentile(df[label], [25, 50, 75])  # Calculate quartiles
            values[i + 1][j + 1] = [round(q, 3) for q in quartiles]  # Round quartiles to 3 decimal places and assign to values matrix


    #print(values)
    # for row in values:
    #     print(row)

    new_numeric = [values[0]]
    for row in values[1:]:
        metric = [row[0]]
        first = [x[0] for x in row[1:]]
        first_row = metric
        for x in first:
            first_row.append(x)
        new_numeric.append(first_row)

        metric = [row[0]]
        med = [x[1] for x in row[1:]]
        med_row = metric
        for x in med:
            med_row.append(x)
        new_numeric.append(med_row)

        metric = [row[0]]
        third = [x[2] for x in row[1:]]
        third_row = metric
        for x in third:
            third_row.append(x)
        new_numeric.append(third_row)


    # print()
    # print(new_numeric)

    import csv

    # Specify the output file name
    output_file = "numeric.csv"

    # Open the output file in write mode
    with open(output_file, mode='w', newline='') as file:
        writer = csv.writer(file)

        # Write each row of the values matrix to the CSV file
        for row in new_numeric:
            writer.writerow(row)

def numeric_basic(filename):
    df = pd.read_csv(filename)
    df = df.set_index("filename")

    labels = ['angular_resolution', 'aspect_ratio', 'crossing_angle', 'edge_crossings', 'edge_length',
            'edge_orthogonality', 'gabriel_ratio', 'neighbourhood_preservation', 'node_resolution',
            'node_uniformity' ]
    
    for label in labels:

        max_value = df[label].max()

        # Find the minimum value in the 'ColumnName' column
        min_value = df[label].min()

        print(f"{label} Minimum value: {min_value}")
        print(f"{label} Maximum value: {max_value}")


def count_drawings(filename):
    df = pd.read_csv(filename)
    df = df.set_index("filename")



    # df = df[~df.iloc[:, 1:].duplicated(keep='first')]
    df_no_hola_circ = df[~df['layout'].isin(["HOLA", "Circular"])]
    # print(df_no_hola_circ.head())
    df_hola_circ = df[df['layout'].isin(["HOLA", "Circular"])]
    #print(len(df_hola_circ))

    df_no_duplicates = df_no_hola_circ[~df_no_hola_circ.iloc[:, 1:].duplicated(keep='first')]

    duplicate_rows = df_no_hola_circ[df_no_hola_circ.iloc[:, 1:].duplicated(keep='first')]

    # Get the count of duplicate rows
    num_duplicate_rows = len(duplicate_rows)

    #print(f"Num duplicates: {num_duplicate_rows}")

    df = pd.concat([df_hola_circ, df_no_duplicates])

    fr_df = df[df["layout"] == "Fruchterman-Reingold"]
    kk_df = df[df["layout"] == "Kamada-Kawai"]
    dr_df = df[df["layout"] == "DRGraph"]
    sugi_df = df[df["layout"] == "Sugiyama"]
    hola_df = df[df["layout"] == "HOLA"]
    circ_df = df[df["layout"] == "Circular"]
    ran_df = df[df["layout"] == "Random"]
    all_df = df[df["layout"] != "Random"]

    generated_df = all_df.copy()
    generated_df = generated_df[~generated_df['generator'].isin(['North', 'Rome'])]

    rome_df = df[df["generator"] == "Rome"]
    north_df = df[df["generator"] == "North"]

    rome_no_ran_df = all_df[all_df["generator"] == "Rome"]
    north_no_ran_df = all_df[all_df["generator"] == "North"]

    dfs = {
        "All": df,
        "Generated": generated_df,
        "Fruchterman-Reingold": fr_df,
        "Kamada-Kawai": kk_df,
        "DRGraph": dr_df,
        "Sugiyama": sugi_df,
        "HOLA": hola_df,
        "Circular": circ_df,
        "Random": ran_df,
        "All (excl. Random)": all_df,
        "Rome": rome_df,
        "North": north_df,
        "Rome (Excl. Random)": rome_no_ran_df,
        "North (Excl. Ranodm)": north_no_ran_df
    }

    for name, dfx in dfs.items():
        print(f"{name}: {dfx.shape[0]} drawings")

    new_dfs = {
        "All": df,
        "Fruchterman-Reingold": fr_df,
        "Kamada-Kawai": kk_df,
        "DRGraph": dr_df,
        "Sugiyama": sugi_df,
        "HOLA": hola_df,
        "Circular": circ_df,
        "Random": ran_df,
        "All (excl. Random)": all_df,
        "Rome": rome_df,
        "North": north_df,
        "Rome (Excl. Random)": rome_no_ran_df,
        "North (Excl. Ranodm)": north_no_ran_df
    }


    


def main():
    # combine_csv()
    #filename = "..\\Data\\metric_example2.csv"
    #filename = "..\\Data\\Graph_Drawing_Metrics_All - Copy.csv"
    #filename = "..\\Data\\Graph_Drawing_Metrics_All_with_NP - Copy.csv"
    #filename = "..\\Data\\Graph_Drawing_Metrics_All_with_NP.csv"
    filename = "Graph_Drawing_Metrics.csv"

    #violin_one_fig_transpose_split_none(filename)
    correlation_matrix(filename)
    # violin_one_fig_transpose_split_rome_north(filename)
    #numeric_distribution(filename)
    #numeric_basic(filename)
    # correlations_same_fig(filename)
    # count_drawings(filename)





    #numeric_distribution(filename)
    # correlation_matrix(filename)
    # correlations_same_fig(filename)
    #violin_one_fig_transpose(filename)
    #violin_one_fig_transpose_split(filename)

    #violin_one_fig_transpose_split_none(filename)
    # violin_one_fig_transpose_split_north(filename)
    # violin_one_fig_transpose_split_rome(filename)
    #correlation_matrix(filename)
    #correlations_same_fig(filename)
    # numeric_distribution(filename)

    #violin_one_fig_transpose(filename)
    #violin_one_fig_transpose_split_rome_north(filename)
    
    #violin_one_fig_transpose_generator(filename)
    #violin_plots_individual(filename)

    #violin_one_fig_transpose_nn()

    
    
    #filename = "..\\Data\\Graph_Properties.csv"
    #violin_plots_individual(filename)
    #violin_one_fig(filename)
    #violin_one_fig_vert(filename)
    
    #violin_one_fig_val(filename)
    #violin_one_fig(filename)
   
    #correlation_matrix(filename)

    #get_distributions(filename)
    #correlation_matrix(filename)
    #get_correlations(filename, one_fig=True)
    
    #corr_matrix_sep2(filename)
    #get_distributions_sep(filename, bins=50)
    #show_correlation_each_graph_type(filename)

    #box_plots(filename)
    #violin_plots(filename)
    #violin_plots_nodes(filename)
    #violin_plots(filename, "BBA_i0_n70_m136.graphml")
    # violin_plots(filename, "ER_i0_n20_m33_p0.162.graphml")
    #violin_plots(filename, "LFR_i1_n80_m172_c10.graphml")
    #violin_plots_nodes_individual(filename, "BBA")
    # violin_plots(filename, "NWS_i1_n30_m67_p0.186_k5.graphml")
    # violin_plots(filename, "g.10.4.graphml")
    # violin_plots(filename, "grafo123.21.graphml")
    #test1(filename)


def combine_csv():

    file1 = 'Graph_Drawing_Metrics_no_circ.csv'
    file2 = 'Graph_Drawing_Metrics_circ.csv'

    # Read the CSV files into DataFrames
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)

    # Concatenate the DataFrames vertically
    merged_df = pd.concat([df1, df2], ignore_index=True)

    # Replace 'merged_file.csv' with the desired name for your merged CSV file
    merged_df.to_csv('Graph_Drawing_Metrics.csv', index=False)


if __name__ == "__main__":
    #combine_csv()
    main()
    