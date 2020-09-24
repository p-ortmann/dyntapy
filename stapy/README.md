# Introduction to STAPY - Static Traffic Assignment for Python 
___
This package has been developed to be used in education and research for traffic management. Once finished, it will 
contain a fair number of commonly used traffic assignment methodologies and their respective components. For background
on the methodologies that are provided here we recommend reading the books provided by Cascetta such as
Transportation Systems Analysis (2009) and more recent Transport Network Analysis (Boyles, 2020). 
___
Development notes and insights (to be edited)

The workflow for StaPy is to first obtain network data in networkx; sources that are supported at this point are osm data  
(integrated through osmnx) and visum (still experimental and limited). Demand is generated randomly, but we're currently
implementing a methodology that allows extraction of crude demand estimates from OSM Data (ETA Q3/Q4 2020).
It is well known that algorithms implemented in pure python can be inherently slow
due to dynamic typing, that's why scientific packages such as Numpy and Scipy thrive. They offer fast
implementations of commonly used algorithms with simple interfaces. Unfortunately, most of the algorithms used in the
traffic assignment field are rather novel and need to be implemented from scratch, to do so we turn to Numba as a
way to circumvent pythons interpreter and keep our code reasonably fast. Numba is a Just-In-Time (JIT) Compiler for
python, this means that when a function is called for the first time with specific inputs a compiled function 
(machinecode) is generated based on the input types. At a later stage when the same function is called again with the 
same inputs that compiled function is retrieved from Cache and runs at a speed that should be close to what is 
attainable in C++. There are some pitfalls to this approach (debugging, black box, ..) but in essence it allows us to code in python without having 
deal with C++/Fortran while still getting the performance benefits associated with these lower level languages, 
for more details check out their website: https://numba.pydata.org/.  
As an indicator for how large the
differences can be we recommend this article https://www.timlrx.com/2019/05/05/benchmark-of-popular-graph-network
-packages/ which shows the differences in calculation times for different implementations of Dijkstra's algorithm
for shortest paths.
Internally, we transform the dictionary structured data from networkx to vectors, lists and matrices in the different 
objects that are supported by Numba. 
One may ask why we use NetworkX then? There are a few reasons (i) The data structures are easy to access in a pythonic way
(ii) it has a lot of useful import/ export tools (iii)  There is a large user base and one can find many packages that 
are build upon it (osmnx) or have examples of visualizations for graphs as they are defined in NetworkX (bokeh, 
holoview, plotly) 

## Getting Started

### Prerequisites

What things you need to install the software and how to install them

```
Give examples
```

### Installing

First download miniconda for your system at: https://docs.conda.io/en/latest/miniconda.html
```
Give the example
```

And repeat

```
until finished
```

End with an example of getting some data out of the system or using it for a little demo

## Running the tests

Explain how to run the automated tests for this system

### Break down into end to end tests

Explain what these tests test and why

```
Give an example
```

### And coding style tests

Explain what these tests test and why

```
Give an example
```

## Deployment

Add additional notes about how to deploy this on a live system

## Built With


## Contributing


## Versioning


## Authors

* **Paul Ortmann** - *Initial work* - [p-ortmann](https://github.com/p-ortmann)


## License

This project is licensed under the GNU GENERAL PUBLIC LICENSE Version 3, 29 June 2007, see license.txt file for details

## Acknowledgments

* Hat tip to anyone whose code was used
* Inspiration 
* etc
