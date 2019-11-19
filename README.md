# Spatial-Temporal Aggregated Predictor Models Implemented in Rcpp 
<!-- badges: start -->
[![Lifecycle: experimental](https://img.shields.io/badge/lifecycle-experimental-orange.svg)](https://www.tidyverse.org/lifecycle/#experimental)
<!-- badges: end -->


## About

This is an R package that fits spatial temporal aggregated predictor models or [STAP](https://arxiv.org/abs/1812.10208) models
using [Rcpp](http://www.rcpp.org/) directly as opposed to stan in the original [rstap](https://biostatistics4socialimpact.github.io/rstap/) package.
This is in order to faciliate faster model fitting as the STAP models' contain nested functions which can result in 
computationally intensive gradient calculations when automated. 

rstap2 currently contains a few toy functions to demonstrate the difference in speed when gradients are hard coded.
I'm still deciding how much further development is warranted.


## Installation


#### Development Version

rstap2 is currently only available as an experimental development package. It can be installed 
in R via the following command.

```r
if(!require(devtools)){
	install.packages("devtools")
	library(devtools)
}
install_github("biostatistics4socialimpact/rstap2")
```


## Acknowledgments 

This work was developed with support from NIH grant R01-HL131610 (PI: Sanchez).
