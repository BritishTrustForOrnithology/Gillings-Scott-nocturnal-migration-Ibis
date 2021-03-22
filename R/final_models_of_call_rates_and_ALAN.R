# models of call rates of thrushes in relation to date and light intensity at each site
# as used in Gillings & Scott 2021 Nocturnal flight calling behaviour of thrushes in relation to artificial light at night. Ibis
# Written by: Simon Gillings
# Last edited: March 2021

#libs for data manipulation
library(tidyr)
library(lubridate)

#libs for modelling
library(mgcv)
library(MASS)
library(ape)

#libs for plots
library(ggplot2)
library(ggthemes)
library(cowplot)

#get the recording effort data
effort <- read.csv(file = 'results/opendata_effort.csv', stringsAsFactors = FALSE )
effort$rec_date <- ymd(effort$rec_date)
effort$start_time_uct <- as.POSIXct(effort$start_time_uct, format = '%Y-%m-%d %H:%M:%S')

#get the detections data
hits <- read.csv(file = 'results/opendata_detections.csv', stringsAsFactors = FALSE)
hits$rec_date <- ymd(hits$rec_date)
hits$passtime <- as.POSIXct(hits$passtime, format = '%Y-%m-%d %H:%M:%S')


#get the siteinfo and covariates
siteinfo <- read.csv(file = 'results/opendata_siteinfo.csv', stringsAsFactors = FALSE)


#remove sites F and W where the recorders developed faults
effort <- subset(effort, !sitecode %in% c('F', 'W'))
hits <- subset(hits, !sitecode %in% c('F', 'W'))

#remove dusk and dawn when detection performance is problematic
effort$hour <- hour(effort$start_time_uct)
effort <- subset(effort, !hour %in% c(17, 5))
hits$hour <- hour(hits$passtime)
hits <- subset(hits, !hour %in% c(17, 5))

#check which light levels are sampled through the study period
#get unique site * rec_date list
site_by_night <- unique(effort[,c('sitecode', 'rec_date')])
site_by_night <- merge(site_by_night, siteinfo, by = 'sitecode')
fig_s2 <- ggplot() +
  geom_point(data = site_by_night, aes(x = (rec_date), y = log(light_mean))) +
  labs(x = 'Recording night date', y = 'Light intensity') + 
  theme_bto_graph() 
fig_s2file <- paste0('results/Fig S2 sampled light through study season.png')
ggsave(fig_s2, file = fig_s2file, width = 6, height = 4, units = 'in')



#get nightly calling activity per site
calls_per_night <- setNames(aggregate(data = hits, hour ~ sitecode + rec_date + species, NROW), c('sitecode', 'rec_date', 'species', 'ncalls'))
head(calls_per_night)

#get nightly total effort per site
effort_per_night <- setNames(aggregate(data = effort, duration ~ sitecode + rec_date, sum), c('sitecode', 'rec_date', 'duration'))
head(effort_per_night)

#get call rates per species and merge into final datasets
#mergeing with effort allows adding of zeroes for when species not detected
cph_b_ <- merge(effort_per_night, calls_per_night[calls_per_night$species=='Blackbird',], by = c('sitecode', 'rec_date'), all.x = TRUE)
cph_b_$species <- 'Blackbird'
cph_re <- merge(effort_per_night, calls_per_night[calls_per_night$species=='Redwing',], by = c('sitecode', 'rec_date'), all.x = TRUE)
cph_re$species <- 'Redwing'
cph_st <- merge(effort_per_night, calls_per_night[calls_per_night$species=='Song Thrush',], by = c('sitecode', 'rec_date'), all.x = TRUE)
cph_st$species <- 'Song Thrush'
cph <- rbind(cph_b_, cph_re, cph_st)
#infill NAs (effort but no detections) with zeroes
cph$ncalls[is.na(cph$ncalls)] <- 0

#add siteinfo to get covariates
cph <- merge(cph, siteinfo, by = 'sitecode')

#create a numeric day variable for modelling
cph$daynum <- as.numeric(cph$rec_date)

#convert loc to factor for modelling random effect of site
cph$loc <- factor(cph$sitecode)





######### Blackbird MODELS #########################################################################################
spp <- 'Blackbird'
onesp_b <- cph[cph$species=='Blackbird',]
#log transform light
onesp_b$light_mean <- log(onesp_b$light_mean)

#model with random site effects
mod_site_b <- gam(data = onesp_b, ncalls ~ s(daynum, k = 4) + s(light_mean, k = 3) + s(loc, bs='re'), offset = log(duration), family = 'poisson')
summary(mod_site_b)
AIC(mod_site_b)

#repeat with linear light term
mod_site_b_linear <- gam(data = onesp_b, ncalls ~ s(daynum, k = 4) + light_mean + s(loc, bs='re'), offset = log(duration), family = 'poisson')
summary(mod_site_b_linear)
plot(mod_site_b_linear, pages=0, select = 1)
AIC(mod_site_b_linear)
#get 95% CI for light effect
as.numeric(summary(mod_site_b_linear)$p.coeff[2] - 1.96*summary(mod_site_b_linear)$se[2])
as.numeric(summary(mod_site_b_linear)$p.coeff[2] + 1.96*summary(mod_site_b_linear)$se[2])

#repeat with linear light term and dropping site S (the site with highest light)
mod_site_b_linear_noS <- gam(data = onesp_b[onesp_b$sitecode != 'S',], ncalls ~ s(daynum, k = 4) + light_mean + s(loc, bs='re'), offset = log(duration), family = 'poisson')
summary(mod_site_b_linear_noS)
plot(mod_site_b_linear_noS, pages=0, select = 1)
AIC(mod_site_b_linear_noS)
#get 95% CI for light effect
as.numeric(summary(mod_site_b_linear_noS)$p.coeff[2] - 1.96*summary(mod_site_b_linear_noS)$se[2])
as.numeric(summary(mod_site_b_linear_noS)$p.coeff[2] + 1.96*summary(mod_site_b_linear_noS)$se[2])

#check residuals for spatial pattern
onesp_b$resids <- residuals.gam(mod_site_b_linear)
dists <- as.matrix(dist(cbind(onesp_b$lon, onesp_b$lat)))
dists.inv <- 1/dists
diag(dists.inv) <- 0
dists.inv[is.infinite(dists.inv)] <- 0
Moran.I(onesp_b$resids, dists.inv)
#if the P value is ns, this means the data are essentially random


######### REDWING MODELS #########################################################################################
spp <- 'Redwing'
onesp_re <- cph[cph$species=='Redwing',]
#log transform light
onesp_re$light_mean <- log(onesp_re$light_mean)

#model with random site effects
mod_site_re <- gam(data = onesp_re, ncalls ~ s(daynum, k = 4) + s(light_mean, k = 3) + s(loc, bs='re'), offset = log(duration), family = 'poisson')
summary(mod_site_re)
AIC(mod_site_re)
plot(mod_site_re, pages=0, select = 1)
plot(mod_site_re, pages=0, select = 2)

#repeat with linear light term
mod_site_re_linear <- gam(data = onesp_re, ncalls ~ s(daynum, k = 4) + light_mean + s(loc, bs='re'), offset = log(duration), family = 'poisson')
summary(mod_site_re_linear)
plot(mod_site_re_linear, pages=0, select = 1)
AIC(mod_site_re_linear)
#get 95% CI for light effect
as.numeric(summary(mod_site_re_linear)$p.coeff[2] - 1.96*summary(mod_site_re_linear)$se[2])
as.numeric(summary(mod_site_re_linear)$p.coeff[2] + 1.96*summary(mod_site_re_linear)$se[2])

#repeat with linear light term and dropping site S (the site with highest light)
mod_site_re_linear_noS <- gam(data = onesp_re[onesp_re$sitecode != 'S',], ncalls ~ s(daynum, k = 4) + light_mean + s(loc, bs='re'), offset = log(duration), family = 'poisson')
summary(mod_site_re_linear_noS)
plot(mod_site_re_linear_noS, pages=0, select = 1)
AIC(mod_site_re_linear_noS)
#get 95% CI for light effect
as.numeric(summary(mod_site_re_linear_noS)$p.coeff[2] - 1.96*summary(mod_site_re_linear_noS)$se[2])
as.numeric(summary(mod_site_re_linear_noS)$p.coeff[2] + 1.96*summary(mod_site_re_linear_noS)$se[2])

#check residuals for spatial pattern
onesp_re$resids <- residuals.gam(mod_site_re_linear)
dists <- as.matrix(dist(cbind(onesp_re$lon, onesp_re$lat)))
dists.inv <- 1/dists
diag(dists.inv) <- 0
dists.inv[is.infinite(dists.inv)] <- 0
Moran.I(onesp_re$resids, dists.inv)
#if the P value is ns, this means the data are essentially random

######### SONG THRUSH MODELS #########################################################################################
spp <- 'Song Thrush'
onesp_st <- cph[cph$species=='Song Thrush',]
#log transform light
onesp_st$light_mean <- log(onesp_st$light_mean)

#model with random site effects
mod_site_st <- gam(data = onesp_st, ncalls ~ s(daynum, k = 4) + s(light_mean, k = 3) + s(loc, bs='re'), offset = log(duration), family = 'poisson')
summary(mod_site_st)
AIC(mod_site_st)
plot(mod_site_st, pages=0, select = 1)
plot(mod_site_st, pages=0, select = 2)

#repeat with linear light term
mod_site_st_linear <- gam(data = onesp_st, ncalls ~ s(daynum, k = 4) + light_mean + s(loc, bs='re'), offset = log(duration), family = 'poisson')
summary(mod_site_st_linear)
plot(mod_site_st_linear, pages=0, select = 1)
AIC(mod_site_st_linear)
#get 95% CI for light effect
as.numeric(summary(mod_site_st_linear)$p.coeff[2] - 1.96*summary(mod_site_st_linear)$se[2])
as.numeric(summary(mod_site_st_linear)$p.coeff[2] + 1.96*summary(mod_site_st_linear)$se[2])

#repeat with linear light term and dropping site S (the site with highest light)
mod_site_st_linear_noS <- gam(data = onesp_st[onesp_st$sitecode != 'S',], ncalls ~ s(daynum, k = 4) + light_mean + s(loc, bs='re'), offset = log(duration), family = 'poisson')
summary(mod_site_st_linear_noS)
plot(mod_site_st_linear_noS, pages=0, select = 1)
AIC(mod_site_st_linear_noS)
#get 95% CI for light effect
as.numeric(summary(mod_site_st_linear_noS)$p.coeff[2] - 1.96*summary(mod_site_st_linear_noS)$se[2])
as.numeric(summary(mod_site_st_linear_noS)$p.coeff[2] + 1.96*summary(mod_site_st_linear_noS)$se[2])

#check residuals for spatial pattern
onesp_st$resids <- residuals.gam(mod_site_st_linear)
dists <- as.matrix(dist(cbind(onesp_st$lon, onesp_st$lat)))
dists.inv <- 1/dists
diag(dists.inv) <- 0
dists.inv[is.infinite(dists.inv)] <- 0
Moran.I(onesp_st$resids, dists.inv)
#if the P value is ns, this means the data are essentially random


# Make final graphs for paper ##########################################################################################

#make new data for predictions
light <- seq(min(cph$light_mean), max(cph$light_mean), length.out = 50)
daynum <- seq(min(cph$daynum), max(cph$daynum), length.out = 50)
#loc <- factor('CB225EX', levels=levels(cph$loc))
loc <- factor('A', levels=levels(cph$loc))
newd <- expand.grid(light_mean = log(light), daynum = daynum, loc = loc)

#make plots with CI
#' @description Produce a pair of plots showing the relationship with 95% confidence limits
#' @param amodel = pass in a model to plot from
#' @param colour = what colour to use for the plot
#' @param maxy1 = maximum for y axis on first plot
#' @param maxy2 = maximum for y axis on second plot
makeplot2 <- function(amodel, colour, maxy1, maxy2) {
  ilink <- family(amodel)$linkinv
  #use lpmatrix method to get predictions per Wood
  preds <- predict(amodel, newdata = newd, type = 'lpmatrix', exclude = s(loc))
  coefs <- coef(amodel)
  vc <- vcov(amodel)
  sim <- mvrnorm(1000, mu = coefs, Sigma = vc)

  #get info for daynum graph
  want_day <- grep("daynum", colnames(preds))
  fits_day <- preds[, want_day] %*% t(sim[, want_day])
  fits_day <- ilink(fits_day)
  
  #now get median and cis for later plotting
  tograph_day <- apply(fits_day, 1, quantile, probs = c(0.025, 0.5, 0.975))
  tograph_day <- as.data.frame(t(tograph_day))  
  tograph_day$daynum <- newd$daynum
  names(tograph_day) <- c('lower', 'median', 'upper', 'daynum')
  tograph_day$date <- as_date(tograph_day$daynum)
  
  #graph1
  graph_day <- ggplot(data=tograph_day, aes(x=date, y = median)) +
    geom_line(col = colour) + 
    geom_ribbon(aes(ymin = lower, ymax = upper), fill = colour, alpha = 0.2, show.legend = FALSE) +
    labs(x='Recording night date', y='Call rate') +
    coord_cartesian(ylim = c(0,maxy1)) +
    theme_bto_graph() +
    theme(plot.margin = unit(c(0.75,0.25,0.25,0.25), "cm")) 

  #get info for light graph
  want_light <- grep("light", colnames(preds))
  fits_light <- preds[, want_light] %*% t(sim[, want_light])
  fits_light <- ilink(fits_light)
  #now get median and cis for later plotting  
  tograph_light <- apply(fits_light, 1, quantile, probs = c(0.025, 0.5, 0.975))
  tograph_light <- as.data.frame(t(tograph_light))  
  tograph_light$daynum <- newd$light
  names(tograph_light) <- c('lower', 'median', 'upper', 'light')
  
  #print ratio of min and max predicted call rate
  cat('Call rate is this much higher at max vs min light intensity:\n')
  print(tograph_light$median[nrow(tograph_light)] / tograph_light$median[1])
  
  
  #graph2
  graph_light <- ggplot(data=tograph_light, aes(x=light, y = median)) +
    geom_line(col = colour) + 
    geom_ribbon(aes(ymin = lower, ymax = upper), fill = colour, alpha = 0.2, show.legend = FALSE) +
    labs(x = "Light intensity", y='Call rate') +
    coord_cartesian(ylim = c(0,maxy2)) +
    theme_bto_graph() +
    theme(plot.margin = unit(c(0.75,0.25,0.25,0.25), "cm")) 

  #layout
  pg <- plot_grid(graph_day, graph_light, nrow = 1)
  
  return(pg)  
  
}

#colours for plots
col_B <- 'Black'
col_RE <- '#CC3300'
col_ST <- '#998c59'

#produce colour figure for paper
plot_b <- makeplot2(amodel = mod_site_b_linear, colour = col_B, maxy1 = 4.5, maxy2 = 8)
plot_re <- makeplot2(mod_site_re_linear, colour = col_RE, maxy1 = 3, maxy2 = 7)
plot_st <- makeplot2(mod_site_st_linear, colour = col_ST, maxy1 = 3, maxy2 = 6)
bigfig <- plot_grid(plot_b, plot_re, plot_st, nrow = 3) +
  annotate("text",x=0.01,y=0.98,size=4,label="(a)", hjust = 0) +
  annotate("text",x=0.01,y=0.65,size=4,label="(b)", hjust = 0) +
  annotate("text",x=0.01,y=0.32,size=4,label="(c)", hjust = 0)
outfile_colour_pdf <- paste0('results/figure_6_colour.pdf')
ggsave(bigfig, file = outfile_colour_pdf, width = 180, height = 160, units = 'mm', scale = 1)



#produce mono figure for paper
plot_b <- makeplot2(amodel = mod_site_b_linear, colour = 'black', maxy1 = 4.5, maxy2 = 8)
plot_re <- makeplot2(mod_site_re_linear, colour = 'black', maxy1 = 3, maxy2 = 7)
plot_st <- makeplot2(mod_site_st_linear, colour = 'black', maxy1 = 3, maxy2 = 6)
bigfig <- plot_grid(plot_b, plot_re, plot_st, nrow = 3) +
  annotate("text",x=0.01,y=0.98,size=4,label="(a)", hjust = 0) +
  annotate("text",x=0.01,y=0.65,size=4,label="(b)", hjust = 0) +
  annotate("text",x=0.01,y=0.32,size=4,label="(c)", hjust = 0)
outfile__mono_pdf <- paste0('results/figure_6_mono.pdf')
ggsave(bigfig, file = outfile__mono_pdf, width = 180, height = 160, units = 'mm', scale = 1)




