"""
Dependence structure discovery
"""
from cdt.causality.graph import PC, GES, GIES
from cdt.independence.graph import ARD, Glasso, DecisionTreeRegression, FSGNN, HSICLasso, LinearSVRL2
import networkx as nx
from data_utils import AbstractDataset
import matplotlib.pyplot as plt


class StructureDiscovery:
    def __init__(self, dataset):
        if isinstance(dataset, AbstractDataset):
            self.dataset = dataset.to_dataframe()
        else:
            raise ValueError("Dataset type not supported.")

        self.output_graph = None  # learned dependence graph

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
        elif method == "ARD":
            obj = ARD()
            self.output_graph = obj.predict(self.dataset)
            neighbors = list(self.output_graph.adj["LABEL"])
        elif method == "Glasso":
            obj = Glasso()
            self.output_graph = obj.predict(self.dataset, alpha=alpha)
            neighbors = list(self.output_graph.adj["LABEL"])
        elif method == "DT":
            obj = DecisionTreeRegression()
            self.output_graph = obj.predict(self.dataset)
            neighbors = list(self.output_graph.adj["LABEL"])
        elif method == "FSGNN":
            obj = FSGNN()
            self.output_graph = obj.predict(self.dataset, threshold=alpha)
            neighbors = list(self.output_graph.adj["LABEL"])
        elif method == "LinearSVRL2":
            obj = LinearSVRL2()
            self.output_graph = obj.predict(self.dataset, threshold=alpha)
            neighbors = list(self.output_graph.adj["LABEL"])
        elif method == "HSICLasso":
            obj = HSICLasso()
            self.output_graph = obj.predict(self.dataset, threshold=alpha)
            neighbors = list(self.output_graph.adj["LABEL"])
        else:
            raise ValueError(f"CSD method {method} not implemented.")

        if display:
            nx.draw_networkx(self.output_graph, font_size=8)
            plt.show()

        return neighbors





    








