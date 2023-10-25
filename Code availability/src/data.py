import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

from sklearn.linear_model import ElasticNetCV, Lasso, LassoCV, MultiTaskElasticNetCV, MultiTaskLassoCV, Ridge, RidgeCV
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, KFold, RepeatedKFold, GridSearchCV, \
    cross_validate, train_test_split
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor


def get_data(recipe_indexes, df):
    rec_cols = []
    for i in recipe_indexes:
        rec_cols += [f"r_{i}", f"g_{i}", f"b_{i}"]
    X = df[rec_cols].values
    Y = df[["conc_water", "conc_co2", "conc_nh3"]].values
    return X, Y


def get_predictions(recipe_indexes, random_state=1, return_model = False, model_type = Ridge, **kwargs):
    # print(recipe_indexes)
    rec_cols = []
    for i in recipe_indexes:
        rec_cols += [f"r_{i}", f"g_{i}", f"b_{i}"]

    df = kwargs.pop("DF")
    X = df[rec_cols].values
    Y = df[["conc_water", "conc_co2", "conc_nh3"]].values
    x_train,x_test,y_train,y_test=train_test_split(X,Y, test_size=0.3,random_state=random_state)
    # alpha = kwargs.get("alpha", 0.02)
    model = model_type(**kwargs)
    # model = Ridge(alpha=alpha, random_state=random_state)
    # model = RidgeCV()
    # model = Lasso(alpha=alpha, random_state=random_state)
    # model = MultiTaskElasticNetCV()
    # model = MultiTaskLassoCV(max_iter=15000)
    # model = MLPRegressor(
    #     hidden_layer_sizes=(),
    #     alpha=0.02,
    #     random_state=1,
    #     max_iter=100000
    #     )
    # model = RandomForestRegressor(n_estimators=25, max_features=10, random_state=1)
    # model = XGBRegressor(
        # objective ='reg:squarederror',
        # # colsample_bytree = 0.3,
        # learning_rate = 0.1,
        # max_depth = 7,
        # alpha = 0.1,
        # n_estimators = 250
    #     )
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    # rmse = mean_squared_error(y_test, y_pred)

    if return_model:
        return recipe_indexes, rmse, model, x_test, y_test, y_pred
    else:
        return recipe_indexes, rmse#, model, x_test, y_test, y_pred


def plot_calibrate_result(recipe_indexes, plot_file_name = "results", model_type = Ridge, save=True, **kwargs):
    res = get_predictions(recipe_indexes, return_model=True, model_type=model_type, **kwargs)
    y = res[4]
    y_pred = res[5]

    fig = plt.figure(figsize=(15,5))
    fig.suptitle(f"{plot_file_name} ({res[1]}) {recipe_indexes}")

    def _set_ax(_ax, desc, _y, _y_pred, unit="%"):
        _ax.scatter(_y, _y_pred)
        xs = np.linspace(min(_y), max(_y), 100)
        _ax.plot(xs, xs, color="k", linestyle="--")
        _ax.axis("equal")
        _ax.set_xlabel(f"Actual {desc} Concentration ({unit})")
        _ax.set_ylabel(f"Calculated {desc} Concentration ({unit})")
        unit_scale = 1e4 if unit == "ppm" else 1
        # _ax.set_title("{} RMSE: {:.2f} {}".format(desc, mean_squared_error(_y, _y_pred) ** 0.5 * unit_scale, unit))
        _ax.set_title("{} error: {:.2f} %".format(desc, mean_squared_error(_y, _y_pred) ** 0.5 / np.mean(_y) * 100))

    ax_1 = fig.add_subplot(131)
    _set_ax(ax_1, "Water", y[:, 0]*200, y_pred[:, 0]*200)

    ax_2 = fig.add_subplot(132)
    _set_ax(ax_2, "CO2", y[:, 1], y_pred[:, 1])

    ax_3 = fig.add_subplot(133)
    _set_ax(ax_3, "NH3", y[:, 2]*0.05 * 1e4, y_pred[:, 2]*0.05 * 1e4, "ppm")

    if save:
        plt.savefig(f"output/{plot_file_name}.png")


def plot_errors(recipe_indexes):
    res = get_predictions(recipe_indexes, return_model=True)
    y = res[4]
    y_pred = res[5]

    fig = plt.figure(figsize=(15,5))
    fig.suptitle(f"Error for {recipe_indexes}", fontsize=16)
    ax = fig.add_subplot(111)
    ax.stem(np.arange(len(y)), (y_pred - y) / y * 100)