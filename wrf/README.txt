PROJECT FOR WORK WITH FILES FROM WRF OUTPUT AND OBSERVED DATA

Structure:

    netcdfUtils: functions do manipulate netcdf files
        --available functions: merge - merge netcdf files
                               getLatLon - return the nearest lat/lon 
                               generateSeries - generate time series for desired coordinate
    
    maps: functions to draw maps
        --available functions: drawMapSeason - draw a map with Season mean
                               drawBreeze - draw a map for each hour in desired range
    
    statistics: funtions to compute and view model performance
        --available functions: generate_metrics - generate sheets with desired metrics, hourly and total
                               generate_mean - generate monthly mean
                               generate_distributions - generate distributions graphs
