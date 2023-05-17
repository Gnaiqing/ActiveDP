"""
Causal structure discovery
"""
import cdt
from cdt.causality.graph import PC, GES
from cdt.data import load_dataset
import networkx as nx
from data_utils import DiscreteDataset, TextDataset, AbstractDataset
import matplotlib.pyplot as plt


class CausalDiscovery:
    def __init__(self, dataset):
        if isinstance(dataset, str):
            self.dataset, self.graph = load_dataset(dataset)
        elif isinstance(dataset, AbstractDataset):
            self.dataset = dataset.to_dataframe()
            self.graph = None
        else:
            raise ValueError("Dataset type not supported.")

        self.output_graph = None

    def causal_structure_discovery(self, method="pc", ci_test="discrete", alpha=0.05, display=True):
        if method == "pc":
            obj = PC(CItest=ci_test, alpha=alpha)
            self.output_graph = obj.predict(self.dataset)
            if display:
                plt.ioff()
                nx.draw_networkx(self.output_graph, font_size=8)
                plt.show()
        elif method == "GES":
            obj = GES(score='int')
            self.output_graph = obj.predict(self.dataset)
            if display:
                plt.ioff()
                nx.draw_networkx(self.output_graph, font_size=8)
                plt.show()

    def display(self):
        if self.output_graph is None:
            print("No output graph available.")
            return

        print("Nodes:", list(self.output_graph.nodes))
        print("Edges:", list(self.output_graph.edges))

    def get_parents(self, v="LABEL"):
        if self.output_graph is None:
            print("No output graph available.")
            return None

        parents = list(self.output_graph.adj[v])
        return parents


    








