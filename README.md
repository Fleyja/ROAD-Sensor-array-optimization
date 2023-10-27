# ROAD-Sensor-array-optimization

# For Data Availability:

We provide data, including gas testing rawData and the data from figures in the manuscript.

The data from figures in the manuscript is included in the files “recipes, t-SNE, array size, errors.xlsx.”

The gas testing data files contain "rawData" in their file names. The "delta_e" worksheet includes the response values of different sensing units at different times. The worksheets "co2_percentage_control," "nh3_percentage_control," and "water_percentage_control" contain the concentration controls for CO2, NH3, and water vapor in mixed gas by MFC, respectively.

In the testing data for the parent array, there are 96 different sensing recipes for CO2, NH3, and water vapor, each prepared with four parallel sensing units. Therefore, the testing data comprises a total of 1152 sensing units. The testing was conducted in three rounds, each with the same gas flow strategy and involving 50 different mixed gas concentrations.

In the calibration testing data, we compared 6 different array evaluation and selection methods. Each method generated a product array of size 10, with each unit prepared in parallel 6 times, resulting in a total of 360 sensing units. The calibration experiments also involved a stepwise change in gas concentration, and exhaust gas bottom needed to be replaced during the experiment, so the calibration testing was conducted in three phases.

Before each gas testing session, a gas mixture with a concentration of 50% of the maximum range is introduced. This procedure aims to flush out any dead volumes in the gas pathways.

The raw data of DBTM process has also been provided.

# For Code Availability

In our ROAD method, the code and data collection processes associated with the first two steps are closely linked to the robotic system. Isolating this portion of the code separately is not feasible without the robotic system. Therefore, we provide only the code for the third step, which includes a genetic algorithm for subarray screening and a backpropagation neural network for array readout.
