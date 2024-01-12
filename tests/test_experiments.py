"""
    test_cvi.py

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

    def test_kmeans(self):
        """
        """
        from src.Kmeans_Clustering_neurogenetic import main
        main(display=False)
