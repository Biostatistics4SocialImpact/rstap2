# Part of the rstap package for estimating model parameters
# 
# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 3
# of the License, or (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.

#' Create a stapreg2 object
#'
#' @param object A list provided by one of the \code{stap_*2} modeling functions.
#' @return A stapreg object
#'
stapreg2 <- function(object,warm_up,iter_max,has_intercept){


	ics <- (warm_up+1):iter_max
	out <- list(alpha = coda::as.mcmc(object$stapfit$alpha_samps[ics,drop=F]),
				delta = coda::as.mcmc(object$stapfit$delta_samps[ics,,drop=F]),
				beta = coda::as.mcmc(object$stapfit$beta_samps[ics,,drop=F]),
				beta_bar = coda::as.mcmc(object$stapfit$beta_bar_samps[ics,,drop=F]),
				theta = coda::as.mcmc(object$stapfit$theta_samps[ics,drop=F]),
				theta_t = coda::as.mcmc(object$stapfit$theta_t_samps[ics,drop=F]),
				sigma = coda::as.mcmc(object$stapfit$sigma_samps[ics,drop=F]),
				sampling_time = object$stapfit$sampling_time,
				call = object$call
				)
    
    structure(out, class = c("stapreg2"))
}




