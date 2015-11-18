#!/usr/bin/env Rscript

## Load packages
library(ggplot2)
library(plyr)

## Read the data set
data.frame = read.csv("table.csv")

## Load labels and labelling functions
source("labels.r")

## Define a function to convert nanoseconds to seconds
nsec2sec = function(nsec)
{
  return(nsec / 1000000000)
}

gradient = function(colors = c("#0000FF", "#11FFFF", "#22FF22", "#FFFF33", "#FF4444", "#FF55FF"), grayscale = c("#FDFDFD", "#020202"), number = 10)
{
	## Steps is a vector of rgb colors the generated gradient must use
	## number is the number of colors to be generated

	segment = function(x, y, i, begin)
	{
		if(begin)
		{
			return(floor((i - 1) / (x - 1) * (y - 1) ) + 1)
		}
		else
		{
			return(ceiling((i - 1) / (x - 1) * (y - 1) ) + 1)
		}
	}

	ratio = function(x, y, i)
	{
		return((i - 1) / (x - 1) * (y - 1) + 1 - segment(x, y, i, TRUE))
	}


	x = number
	y = length(colors)

	yy = length(grayscale)

	out = list()

	for(i in 1:number)
	{
		r = ratio(x, y, i)
		begin = segment(x, y, i, TRUE)
		end = segment(x, y, i, FALSE)
		#cat(paste(sep = "", "i: ", as.character(i), "; x: ", as.character(x), "; y = ", as.character(y), "; begin: ", as.character(begin), "; end: ", as.character(end), "; r: ", as.character(r)))

		rbegin = col2rgb(colors[begin], alpha = TRUE)["red",]
		gbegin = col2rgb(colors[begin], alpha = TRUE)["green",]
		bbegin = col2rgb(colors[begin], alpha = TRUE)["blue",]
		abegin = col2rgb(colors[begin], alpha = TRUE)["alpha",]

		rend = col2rgb(colors[end], alpha = TRUE)["red",]
		gend = col2rgb(colors[end], alpha = TRUE)["green",]
		bend = col2rgb(colors[end], alpha = TRUE)["blue",]
		aend = col2rgb(colors[end], alpha = TRUE)["alpha",]

		#cat(paste(sep="", "rbegin: ", as.character(rbegin), "; gbegin: ", as.character(gbegin), "; bbegin: ", as.character(bbegin), "; abegin: ", as.character(abegin), ".\n"))
		#cat(paste(sep="", "rend: ", as.character(rend), "; gend: ", as.character(gend), "; bend: ", as.character(bend), "; aend: ", as.character(aend), ".\n"))

		## Create the intermediate color
		rout = rbegin * (1 - r) + rend * r
		gout = gbegin * (1 - r) + gend * r
		bout = bbegin * (1 - r) + bend * r
		aout = abegin * (1 - r) + aend * r

		#cat(paste(sep="", "rout: ", as.character(rout), "; gout: ", as.character(gout), "; bout: ", as.character(bout), "; aout: ", as.character(aout), ".\n"))

		rr = ratio(x, yy, i)
		gbegin = segment(x, yy, i, TRUE)
		gend = segment(x, yy, i, FALSE)
		#cat(paste(sep = "", "i: ", as.character(i), "; x: ", as.character(x), "; yy = ", as.character(yy), "; gbegin: ", as.character(gbegin), "; gend: ", as.character(gend), "; rr: ", as.character(rr)))

		## Integrate luminosity into the gradient, for grayscale outputs
		hsvbegin = rgb2hsv(r = col2rgb(grayscale[gbegin], alpha = TRUE)["red",], g = col2rgb(grayscale[gbegin], alpha = TRUE)["green",], b = col2rgb(grayscale[gbegin], alpha = TRUE)["blue",], maxColorValue = 255)
		hsvend = rgb2hsv(r = col2rgb(grayscale[gend], alpha = TRUE)["red",], g = col2rgb(grayscale[gend], alpha = TRUE)["green",], b = col2rgb(grayscale[gend], alpha = TRUE)["blue",], maxColorValue = 255)
		hsvout = rgb2hsv(r = rout, g = gout, b = bout, maxColorValue = 255)

		hsvout[3] = hsvbegin[3] * (1 - rr) + hsvend[3] * rr

		out = append(out, hsv(hsvout[1], hsvout[2], hsvout[3], aout / 255))
		#cat("####################\n\n")
	}

	return(unlist(out))
}

## Compute the execution time for each run of the experiment
data.frame = ddply(
  data.frame, c("measure", "max_push_pop", "nb_threads", "non_blocking", "thread", "try"), summarize,
  thread_time = 
    thread_stop_sec + nsec2sec(thread_stop_nsec) - thread_start_sec - nsec2sec(thread_start_nsec),
  global_time = 
    global_stop_sec + nsec2sec(global_stop_nsec) - global_start_sec - nsec2sec(global_start_nsec)
)

## Compute mean and standard deviation
data.frame = ddply(
  data.frame, c("measure", "max_push_pop", "nb_threads", "non_blocking", "thread"), summarize,
  thread_mean_time = mean(thread_time),
  global_mean_time = mean(global_time),
  thread_time_std = sd(thread_time),
  global_time_std = sd(global_time)
)

## Create a simple plot with ggplots
plot = ggplot() +
  geom_line(data = apply_labels(data.frame[data.frame$measure == 1,]),
            aes(nb_threads, global_mean_time, group = non_blocking, color = non_blocking),
            size = 1) +
  geom_point(data = apply_labels(data.frame[data.frame$measure == 1,]),
            aes(nb_threads, global_mean_time, group = non_blocking, color = non_blocking, shape = non_blocking),
            size = 3) +
  guides(fill = guide_legend(title = "Thread"), colour = guide_legend(title = "Synchronization"), shape = guide_legend(title = "Synchronization")) +
  ylab("Running time in seconds") +
  xlab(label("nb_threads")) +
  ggtitle("Running time for pop operations") + 
  scale_fill_manual(values = gradient(number = 9)) +
  scale_colour_manual(values = gradient(colors = c("#DD0000", "#008800"), grayscale = c("#DD0000", "#008800"), number = 3))
## Save the plot as a svg file
ggsave(file = "global_timing_pop.svg", plot = plot, width = 8, height = 6)

## Create a simple plot with ggplots
plot = ggplot() +
  geom_line(data = apply_labels(data.frame),
            aes(nb_threads, global_mean_time, group = interaction(non_blocking, measure), color = non_blocking),
            size = 1) +
  geom_point(data = apply_labels(data.frame),
            aes(nb_threads, global_mean_time, group = interaction(non_blocking, measure), color = non_blocking, shape = measure),
            size = 3) +
  guides(fill = guide_legend(title = "Thread"), colour = guide_legend(title = "Synchronization"), shape = guide_legend(title = "Operation")) +
  ylab("Running time in seconds") +
  xlab(label("nb_threads")) +
  ggtitle("Running time for push and pop operations") + 
  scale_fill_manual(values = gradient(number = 9)) +
  scale_colour_manual(values = gradient(colors = c("#DD0000", "#008800"), grayscale = c("#DD0000", "#008800"), number = 3))
## Save the plot as a svg file
ggsave(file = "global_timing_both.svg", plot = plot, width = 8, height = 6)

## Create a simple plot with ggplots
plot = ggplot() +
  geom_line(data = apply_labels(data.frame[data.frame$measure == 2,]),
            aes(nb_threads, global_mean_time, group = non_blocking, color = non_blocking),
            size = 1) +
  geom_point(data = apply_labels(data.frame[data.frame$measure == 2,]),
            aes(nb_threads, global_mean_time, group = non_blocking, color = non_blocking, shape = non_blocking),
            size = 3) +
  guides(fill = guide_legend(title = "Thread"), colour = guide_legend(title = "Synchronization"), shape = guide_legend(title = "Synchronization")) +
  ylab("Running time in seconds") +
  xlab(label("nb_threads")) +
  ggtitle("Running time for push operations") + 
  scale_fill_manual(values = gradient(number = 9)) +
  scale_colour_manual(values = gradient(colors = c("#DD0000", "#008800"), grayscale = c("#DD0000", "#008800"), number = 3))
## Save the plot as a svg file
ggsave(file = "global_timing_push.svg", plot = plot, width = 8, height = 6)

