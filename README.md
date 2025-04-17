# Econometrics with Simulations Book

* To create a specific conda environment that will handle required libraries:
``conda env create -f environment.yml`` 
* The simulations folder contains the code of the simulations.
* The plan for each simulation, together with a discussion of the theory, simulation plan, and other details  will be discussed in a specific Github Issue.
* In order to contribute to the project, push to a specific branch. 


# Deployment 

## 1. Dashboards deployment

* Once dashboards are ready for production, they should be added to simulations/launcher.py . This python file automatically deploys the dashboards to specified ports. So from the simulations folder call:
``python launcher.py``

## 2. Book deployment

* If the book has already been built, look for the book in html form at book/_build

* Alternatively, you can build your own development version of the book calling (from the root folder):  
    ``jupyter--book build book``


## Server deployment
* To deploy the html contents of the book using Python’s HTTP Server (Quick Solution): Navigate to the _build/html/ folder and start the server: 

    ``` 
    tmux new -s book

    cd _build/html 
    
    python3 -m http.server 8000 
    ```

* In another session deploy the simulations. Don't forget the --server option! 

    ```
    tmux new -s simus  
    python launcher.py --server
    ```