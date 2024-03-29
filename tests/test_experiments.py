"""
    test_experiments.py

# Description
Runs each experiment within pytest to verify script functionality during CI.

# Authors
- Sasha Petrenko <sap625@mst.edu>
"""

# -----------------------------------------------------------------------------
# TESTS
# -----------------------------------------------------------------------------


class TestExperiments:
    """
    Pytest class containing experiment unit tests.
    """

    def test_shap(self):
        # Include the test code definitions
        from src.SHAP_for_neurogenetic_diseases import main

        # Run the experiment without displaying
        main(display=False)

        return

    def test_centroids(self):
        # Include the test code definitions
        from src.plot_centroids_for_neurogenetic_t_SNE import main

        # Run the experiment without displaying
        main(display=False)

        return

    def test_bar_features(self):
        # Include the test code definitions
        from src.neurogenetic_stacked_bar_features import main

        # Run the experiment without displaying
        main(display=False)

        return

    def test_kmeans(self):
        # Include the test code definitions
        from src.Kmeans_Clustering_neurogenetic import main

        # Run the experiment without displaying
        main(display=False)

        return
