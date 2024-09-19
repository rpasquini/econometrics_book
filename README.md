# Econometrics with Simulations Book

* To create a specific conda environment that will handle required libraries:
``conda env create -f environment.yml`` 
* The simulations folder contains the code of the simulations.
* The plan for each simulation, together with a discussion of the theory, simulation plan, and other details  will be discussed in a specific Github Issue.
* In order to contribute to the project, push to a specific branch. 


# Deployment 

## 1. Dashboards deployment

* Once dashboards are ready for production, they should be added to simulations/launcher.py . This python file automatically deploys the dashboards to specified ports. So from the root call:
``python simulations/launcher.py``

## 2. Book deployment

* In development mode just look for the book in html form in book/_build
* You can build your own development version of the book calling (from the root) 

``jupyter--book build book``
