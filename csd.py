"""
Causal structure discovery
"""
import cdt
import pandas as pd
from cdt.causality.graph import PC, GES, GIES
from cdt.causality.graph.bnlearn import MMPC
from cdt.independence.graph import ARD, Glasso
import networkx as nx
from data_utils import DiscreteDataset, TextDataset, AbstractDataset
import matplotlib.pyplot as plt


class CausalDiscovery:
    def __init__(self, dataset):
        if isinstance(dataset, AbstractDataset):
            self.dataset = dataset.to_dataframe()
            self.graph = None
        else:
            raise ValueError("Dataset type not supported.")

        self.output_graph = None  # learned dependence graph or causal strcture graph

    # def causal_structure_discovery(self, method="PC", ci_test="discrete", alpha=0.05, display=True):
    #     if method == "pc":
    #         obj = PC(CItest=ci_test, alpha=alpha)
    #         self.output_graph = obj.predict(self.dataset)
    #         if display:
    #             plt.ioff()
    #             nx.draw_networkx(self.output_graph, font_size=8)
    #             plt.show()
    #     elif method == "GES":
    #         obj = GES(score='int')
    #         self.output_graph = obj.predict(self.dataset)
    #         if display:
    #             plt.ioff()
    #             nx.draw_networkx(self.output_graph, font_size=8)
    #             plt.show()

    def get_neighbor_nodes(self, method, alpha=0.05, display=True):
        """
        Get neighbor nodes for target variable by causal structure learning
        :param method:
        :param alpha:
        :return:
        """
        if method == "PC":
            obj = PC(CItest="discrete", alpha=alpha)
            self.output_graph = obj.predict(self.dataset)
            H = nx.Graph(self.output_graph)  # convert to undirected graph
            neighbors = list(H.adj["LABEL"])
        elif method == "GES":
            obj = GES(score="int")
            self.output_graph = obj.predict(self.dataset)
            H = nx.Graph(self.output_graph)  # convert to undirected graph
            neighbors = list(H.adj["LABEL"])
        elif method == "GIES":
            obj = GIES(score="int")
            self.output_graph = obj.predict(self.dataset)
            H = nx.Graph(self.output_graph)
            neighbors = list(H.adj["LABEL"])
        elif method == "MMPC":
            # TODO: fix this method
            obj = MMPC(alpha=alpha)
            self.output_graph = obj.predict(self.dataset)
            H = nx.Graph(self.output_graph)
            neighbors = list(H.adj["LABEL"])
        elif method == "ARD":
            obj = ARD()
            self.output_graph = obj.predict(self.dataset)
            neighbors = list(self.output_graph.adj["LABEL"])
            # check the score of each feature
            features = self.dataset.drop(["LABEL"], axis=1)
            labels = self.dataset["LABEL"]
            scores = obj.predict_features(features, labels)
            scores_df = pd.DataFrame({
                "features": features.columns,
                "scores": scores
            })
        elif method == "Glasso":
            obj = Glasso()
            self.output_graph = obj.predict(self.dataset, alpha=alpha)
            neighbors = list(self.output_graph.adj["LABEL"])
        else:
            raise ValueError(f"CSD method {method} not implemented.")

        if display:
            nx.draw_networkx(self.output_graph, font_size=8)
            plt.show()

        return neighbors





    








