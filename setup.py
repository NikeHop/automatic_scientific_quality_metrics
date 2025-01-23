from setuptools import setup, find_packages

setup(
    name="automatic_scientific_quality_metrics",  # Replace with your project name
    version="0.1.0",  # Replace with your project's version
    author="Niklas Hoepner",
    author_email="n.r.hopner@uva.nl",
    description="Code for the paper: Automatic Evaluation Metrics for Artificially Generated Scientific Research",
    python_requires=">=3.11",  # Specify minimum Python version
    packages=["automatic_scientific_qm"], 
)