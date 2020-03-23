
# tkinter imports
from tkinter import *
from tkinter import filedialog
from tkinter.messagebox import showinfo, showerror
from tkinter.ttk import Progressbar, Separator, LabelFrame

# sklearn imports
import sklearn
from sklearn import ensemble
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn import metrics, gaussian_process, neural_network

import xgboost as xgb

# other imports
import os
import time
import pylint
import numpy as np
import pandas as pd
import pickle
import ctypes
import matplotlib.pyplot as plt
import seaborn as sns
from PyQt5 import QtGui
from pandastable import Table, TableModel
import inspect
import webbrowser


__author__ = "Nils Kohring"
__version__ = "0.0.1"



"""
    TODO
    - fix handeling of tuples as model parameters (see eg. for NNs)
    - support for categorical data: use scikit ColumnTransfromer!
    - more models
    - real progressbars
    - where left off
    - having multiple models at same time
    - cross validation

"""

class MainWindow(Frame):
    def __init__(self, master=None):
        Frame.__init__(self, master)
        self.grid()
        self.master.title("Simple Data Modeling")
        self.createWidgets()
        self.model_loaded = False
        self.header_bool = IntVar()

        # define model types (classification/regression)
        self.clas_models = {
            'Decision Tree Classifier': sklearn.tree.DecisionTreeClassifier,
            'Random Forest Classifier':  ensemble.RandomForestClassifier,
            'Gradient Boosting Classifier': sklearn.ensemble.GradientBoostingClassifier
        }
        self.reg_models = {
            'Linear Model': sklearn.linear_model.LinearRegression,
            'k-Nearest Neighbors Regression': sklearn.neighbors.KNeighborsRegressor,
            'Decision Tree Regression':sklearn.tree.DecisionTreeRegressor,
            'Gaussian Process Regression': gaussian_process.GaussianProcessRegressor,
            'Neural Network': neural_network.MLPRegressor,
            'Support Vector Regression':sklearn.svm.SVR,
            'XGB Regressor': xgb.XGBRegressor,
        }

        # define metric types for evaluation (classification/regression)
        self.clas_metrics = {
            "Accuracy Classification Score": metrics.accuracy_score,
            "Balanced Accuracy": metrics.balanced_accuracy_score,
            "F1 Score": metrics.f1_score,
            "Precision": metrics.precision_score,
            "Recall": metrics.recall_score
	    }
        self.reg_metrics = {
            "Mean Absolute Error": metrics.mean_absolute_error,
            "Mean Squared Error": metrics.mean_squared_error,
            "R\u00B2 (Coefficient of Determination)": metrics.r2_score
        }

    def createWidgets(self):
        top = self.winfo_toplevel()
        self.menuBar = Menu(top)
        top["menu"] = self.menuBar
        self.subMenu = Menu(self.menuBar, tearoff=0)
        self.menuBar.add_cascade(label="File", menu=self.subMenu)
        self.subMenu.add_command(label="New", command=self.readData_createWindow)
        self.subMenu.add_command(label="Save As...", command=self.save)
        self.subMenu.add_command(label="Load Model", command=self.laod_model)
        self.subMenu.add_separator()
        self.subMenu.add_command(label="Properties", command=self.properties)
        self.subMenu.add_command(label="Help", command=self.help)
        self.menuBar.add_command(label="Quit", command=self.master.destroy)

    def readData_createWindow(self):
        try:
            filename = filedialog.askopenfilename()
            f = open(filename, "rb")
            # f = 'http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
            self.set_header(self.header_bool)
            header = self.header_bool.get() == 1
            self.data = pd.read_csv(f, header=0 if header else None, sep=',')
            if not header:
                self.data.columns = ['var' + str(i) for i in range(len(self.data.columns))]
        except AttributeError:
            pass
        except Exception as e:
            return showerror("Error", "Data could not be loaded! Make sure the data is " \
                      + "formated in a similar way as the sample data: {}.".format(e))

        self.cols = self.data.columns
        self.setUpData()
        self.bools = []

        height = min(5, self.data.shape[0])
        width = len(self.cols)


        # data frame
        data_frame = LabelFrame(self.master, text="Data Summary", relief=RIDGE)
        data_frame.grid(row=0, column=0, columnspan=5, sticky="EWNS", padx=15, pady=7.5)
        data_size = "Data size is {} bytes. ".format(self.data.memory_usage(deep=True).sum())
        cols_rows = "Number of columns is {} and the number of rows is {}. " \
            .format(self.data.shape[1], self.data.shape[0])
        miss_data = "The data does have missing values." if self.data.isnull().values.any() \
            else "The data does not have missing values."
        Message(data_frame, text=data_size + cols_rows + miss_data, width=315) \
            .grid(row=0, column=0, columnspan=width, rowspan=3, sticky='W')
        Button(data_frame, text="Data", width=12, command=self.show_data) \
               .grid(row=0, column=4, columnspan=1)
        Button(data_frame, text="Description", width=12, command=self.show_description) \
               .grid(row=1, column=4, columnspan=1)
        Button(data_frame, text="Pair Plot", width=12, command=self.pair_plot) \
               .grid(row=2, column=4, columnspan=1)


        # head frame
        head_frame = LabelFrame(self.master, text="Head of data", relief=RIDGE)
        head_frame.grid(row=5, column=0, columnspan=5, sticky="EWNS", padx=15, pady=7.5)
        for i, btn in enumerate(self.cols):
            new_btn = Menubutton(head_frame, text=btn, width=11, relief='raised')
            self.bools.append(self.initBools(5, trues=[0, 3] if i < width-1 else [1, 3]))
            self.setColSelection(new_btn, i)
            new_btn.grid(row=0, column=i, columnspan=1)
        for i in range(height):
            for j in range(width):
                b = Label(head_frame, text=self.data[self.data.columns[j]][i],
                          width=12, relief="groove")
                b.grid(row=1+i, column=j)


        # modeling frame
        model_frame = LabelFrame(self.master, text="Modeling", relief=RIDGE)
        model_frame.grid(row=height+11, column=0, columnspan=5, sticky="EWNS", padx=15, pady=7.5)
        
        # train / test ratio
        self.train_test_ratio = 1/3
        self.train_test_ratio_str = StringVar(
            self.master,
            value = str(np.round(self.train_test_ratio, 2))
        )
        Button(model_frame, text="Shuffle Data", width=12, command=self.shuffle_data) \
               .grid(row=0, column=0, columnspan=1)
        Button(model_frame, text="Train-Test Ratio", width=24, command=self.set_traintest) \
               .grid(row=0, column=1, columnspan=2)
        Label(model_frame, textvariable=self.train_test_ratio_str) \
               .grid(row=0, column=3, columnspan=1, sticky="E")

        # model selection
        self.model_btn = Menubutton(model_frame, text="Model Type", width=12, relief='raised')
        self.set_model(self.model_btn)
        self.model_btn.grid(row=1, column=0, columnspan=1)
        Button(model_frame, text="Parameters", width=12, command=self.set_model_parameters) \
               .grid(row=1, column=1, columnspan=1)
        self.metric_btn = Menubutton(model_frame, text="Metric", width=12, relief='raised')
        self.set_metric(self.metric_btn)
        self.metric_btn.grid(row=1, column=2, columnspan=1)

        # model training
        self.score = -1
        self.score_str = StringVar(self.master, value="")
        Button(model_frame, text="Train Model", width=12, command=self.startModel) \
               .grid(row=1, column=3, columnspan=width-1)
        Label(model_frame, textvariable=self.score_str) \
               .grid(row=3, columnspan=width, sticky='W')

        # Export Frame
        export_frame = LabelFrame(self.master, text="Save", relief=RIDGE)
        export_frame.grid(row=height+15, column=0, columnspan=5, sticky="EWNS", padx=15, pady=7.5)
        Button(export_frame, text="Export Data", width=12, command=self.export_data) \
               .grid(row=0, column=0, columnspan=1)
        Button(export_frame, text="Export Model", width=12, command=self.export_model) \
               .grid(row=0, column=1, columnspan=1)

    def set_header(self, header_var):
        """
        User selection if data has a header as first row.
        """
        class popupWindow(object):
            def __init__(self, master):
                top = self.top = Toplevel(master)
                top.title('Header')
                top.attributes("-topmost", True)
                top.geometry("300x65")
                top.grid()
                top.resizable(0, 0)
                self.e = Checkbutton(top, text="Data contains header", variable=header_var,
                    onvalue=1, offvalue=0)
                self.e.grid(row=0, column=1, columnspan=1)
                Button(top, text='     Ok     ', command=self.cleanup) \
                    .grid(row=0, column=2, columnspan=1)
                top.bind('<Return>', self.cleanup)
            def cleanup(self, event=None):
                self.top.destroy()

        popup = popupWindow(self.master)
        self.master.wait_window(popup.top)

    def setUpData(self):
        self.input_cols = list(np.arange(len(self.cols) - 1))
        self.output_cols = [len(self.cols) - 1]
        self.ignore_cols = []
        self.numeric_cols = list(np.arange(len(self.cols)))
        self.categorical_cols = []

    def setColSelection(self, new_btn, col):
        new_menu = Menu(new_btn, tearoff=0)
        new_btn["menu"] = new_menu
        new_menu.add_checkbutton(label="Input", command=self.setIn(col),
                                 variable=self.bools[col][0], onvalue=1, offvalue=0)#, 
                                 #state='disabled' if col < len(self.cols)-1 else 'active')
        new_menu.add_checkbutton(label="Output", command=self.setOut(col),
                                 variable=self.bools[col][1])
        new_menu.add_checkbutton(label="Ignore", command=self.setIgnore(col),
                                 variable=self.bools[col][2])
        new_menu.add_separator()
        new_menu.add_checkbutton(label="Numeric", command=self.setNum(col),
                                 variable=self.bools[col][3])#, state='disabled')
        new_menu.add_checkbutton(label="Categorical", command=self.setCat(col),
                                 variable=self.bools[col][4])
        return new_menu

    def initBools(self, n, trues=[0, 3]):
        return [BooleanVar(self.master, value=True if i in trues else False)
                for i in range(n)]

    ### functions for feature settings ###
    def setIn(self, col):
        def _setIn():
            if col in self.input_cols:
                pass
            else: 
                self.input_cols.append(col)
                self.input_cols = sorted(self.input_cols)
                try: del self.output_cols[self.output_cols.index(col)]
                except: pass
                try: del self.ignore_cols[self.ignore_cols.index(col)]
                except: pass
            self.bools[col][0].set(value=True)
            self.bools[col][1].set(value=False)
            self.bools[col][2].set(value=False)
        return _setIn

    def setOut(self, col):
        def _setOut():
            if col in self.output_cols:
                pass
            else:
                try: del self.input_cols[self.input_cols.index(col)]
                except: pass
                try: del self.ignore_cols[self.ignore_cols.index(col)]
                except: pass
                if len(self.output_cols) == 1:
                    self.input_cols.append(self.output_cols[0])
                    self.bools[self.output_cols[0]][0].set(value=True)
                    self.bools[self.output_cols[0]][1].set(value=False)
                self.output_cols = [col]
            self.bools[col][0].set(value=False)
            self.bools[col][1].set(value=True)
            self.bools[col][2].set(value=False)
        return _setOut

    def setIgnore(self, col):
        def _setOut():
            if col in self.ignore_cols:
                pass
            else:
                self.ignore_cols.append(col)
                self.ignore_cols = sorted(self.ignore_cols)
                try: del self.input_cols[self.input_cols.index(col)]
                except: pass
                try: del self.output_cols[self.output_cols.index(col)]
                except: pass
            self.bools[col][0].set(value=False)
            self.bools[col][1].set(value=False)
            self.bools[col][2].set(value=True)
        return _setOut

    def setNum(self, col):
        def _setNum():
            if col in self.numeric_cols:
                pass
            else:
                self.numeric_cols.append(col)
                self.numeric_cols = sorted(self.numeric_cols)
                try: del self.categorical_cols[self.categorical_cols.index(col)]
                except: pass
            self.bools[col][3].set(value=True)
            self.bools[col][4].set(value=False)
        return _setNum

    def setCat(self, col):
        def _setCat():
            # showinfo("Information", "Categorical variables not supported yet!")
            if col in self.categorical_cols:
                pass
            else:
                self.categorical_cols.append(col)
                self.categorical_cols = sorted(self.categorical_cols)
                try: del self.numeric_cols[self.numeric_cols.index(col)]
                except: pass
            self.bools[col][3].set(value=False)
            self.bools[col][4].set(value=True)
        return _setCat
    ### ------------------------------ ###

    def save(self):
        pass

    def properties(self):
        pass

    def help(self):
        showinfo("Information", "This Application is for small data modeling tasks.\n\n" \
            + "Authors\t{}\nVersion\t{}".format(__author__, __version__))

    def set_traintest(self):

        class popupWindow(object):
            def __init__(self, master):
                top = self.top = Toplevel(master)
                top.title('Train-Test Ratio')
                top.attributes("-topmost", True)
                top.geometry("350x75")
                top.grid()
                top.resizable(0, 0)
                Label(top, text="Enter the ratio of training set " \
                    + "size to test set size:").grid(row=0, columnspan=6)
                self.e = Entry(top)
                self.e.grid(row=1, column=0, columnspan=5)
                Button(top, text='     Ok     ', command=self.cleanup) \
                    .grid(row=1, column=5, columnspan=1)
                top.bind('<Return>', self.cleanup)
            def cleanup(self, event=None):
                self.value = self.e.get()
                self.top.destroy()

        popup = popupWindow(self.master)
        self.master.wait_window(popup.top)
        try:
            train_test_ratio = float(popup.value)
            if train_test_ratio > 1 or train_test_ratio <= 0:
                showerror("Error", "Train-Test ratio must be in (0,1]!")
            else:
                self.train_test_ratio = train_test_ratio
            self.train_test_ratio_str.set(str(np.round(self.train_test_ratio, 2)))
        except AttributeError:
            pass
        except:
            showerror("Error", "Train-Test ratio must be a value in (0,1]!")
    
    def set_model(self, btn):
        new_menu = Menu(btn, tearoff=0)
        btn["menu"] = new_menu

        self.model_selection_bool = self.initBools(
            len(self.reg_models) + len(self.clas_models), trues=[]
        )

        for i, model in enumerate(self.reg_models.keys()):
            new_menu.add_checkbutton(label=model, command=self.setModelType(i),
                variable=self.model_selection_bool[i])
        new_menu.add_separator()
        for i, model in enumerate(self.clas_models.keys()):
            j = len(self.reg_models) + i
            new_menu.add_checkbutton(
                label=model, command=self.setModelType(j),
                variable = self.model_selection_bool[j]
            )

        return new_menu
    
    def set_metric(self, btn):
        new_menu = Menu(btn, tearoff=0)
        btn["menu"] = new_menu

        self.metric_selection_bool = self.initBools(
            len(self.reg_metrics) + len(self.clas_metrics), trues=[]
        )

        for i, metric in enumerate(self.reg_metrics.keys()):
            new_menu.add_checkbutton(label=metric, command=self.setMetricType(i),
                variable=self.metric_selection_bool[i])
        new_menu.add_separator()
        for i, metric in enumerate(self.clas_metrics.keys()):
            new_menu.add_checkbutton(
                label=metric, command=self.setMetricType(len(self.reg_metrics) + i),
                variable=self.metric_selection_bool[len(self.reg_metrics) + i]
            )

        return new_menu

    def setModelType(self, model_index, default_params=True):
        def _setModelType():
            if len(self.output_cols) == 0:
                showerror("Error", "No output variable selected!")
            if self.output_cols[0] in self.numeric_cols:
                if model_index >= len(self.reg_models): # regression
                    for j in range(len(self.model_selection_bool)):
                        self.model_selection_bool[j].set(value=False)
                    return showerror("Error", "This model is for classification!")
            elif self.output_cols[0] in self.categorical_cols:
                if model_index < len(self.reg_models):
                    for j in range(len(self.model_selection_bool)):
                        self.model_selection_bool[j].set(value=False)
                    return showerror("Error", "This model is for regression!")

            for j in range(len(self.model_selection_bool)):
                self.model_selection_bool[j].set(value=False)
            self.model_selection_bool[model_index].set(value=True)

            # get selected model type
            for i, b in enumerate(self.model_selection_bool):
                if b.get():
                    self.model_int = i

            try:
                if self.output_cols[0] in self.numeric_cols: # regression
                    model = self.reg_models[list(self.reg_models.keys())[self.model_int]]
                    model_name = list(self.reg_models.keys())[self.model_int]
                elif self.output_cols[0] in self.categorical_cols: # classification
                    model_int = self.model_int - len(self.reg_models)
                    model = self.clas_models[list(self.clas_models.keys())[model_int]]
                    model_name = list(self.clas_models.keys())[model_int]

            except Exception as e:
                return showerror("Error", "An appropriate model has to be selected: {}".format(e))

            self.model_name = model_name
            self.model = model

            # get and set default model parameters
            if default_params:
                signature = inspect.signature(model)
                self.model_dict = {
                    k: v.default
                    for k, v in signature.parameters.items()
                    if v.default is not inspect.Parameter.empty
                }

            # transformation for regression tasks
            if self.output_cols[0] in self.numeric_cols:
                def transformedTargetRegressor(**kwargs):
                    return TransformedTargetRegressor(
                            regressor = model(**kwargs),
                            transformer = MinMaxScaler()
                        )
                model = transformedTargetRegressor

        return _setModelType

    def set_model_parameters(self):
        """
        """
        try:
            model_name, model, model_dict = self.model_name, self.model, self.model_dict
        except Exception as e:
            return showerror("Error", "An model has to be selected first: {}".format(e))

        class popupWindow(object):
            def __init__(self, master):
                top = self.top = Toplevel(master)
                top.title("")
                top.attributes("-topmost", True)
                width = str(27 * (len(model_dict) + 2))
                top.geometry("350x" + width)
                top.grid()
                top.resizable(0, 0)
                Label(top, text='Parameters of ' + model_name) \
                    .grid(row=0, column=0, columnspan=2)

                self.values = list()
                line = 1
                for k, v in model_dict.items():
                    Label(top, text=k).grid(row=line, column=0, columnspan=1, sticky='W')
                    e = Entry(top)
                    e.insert(END, str(v))
                    e.grid(row=line, column=1, columnspan=1)
                    self.values.append(e)
                    line += 1

                Button(top, text='  Help  ', command=self.help) \
                    .grid(row=line+2, column=0, columnspan=1, sticky='W')
                Button(top, text='   Ok   ', command=self.cleanup) \
                    .grid(row=line+2, column=1, columnspan=1, sticky='E')
                top.bind('<Return>', self.cleanup)
            def cleanup(self, event=None):
                for i, k in enumerate(model_dict.keys()):
                    param_type = type(model_dict[k])
                    try:
                        value = self.values[i].get()
                        if param_type in (type(tuple()), type(list())):
                            model_dict[k] = eval(value)
                        elif param_type is not type(None):
                            model_dict[k] = param_type(value)
                        elif value == 'True' or value == 'False':
                            model_dict[k] = bool(value)
                        else: # default might be NoneType which we don't want
                            try:
                                model_dict[k] = float(value)
                            except:
                                try:
                                    model_dict[k] = int(value)
                                except:
                                    model_dict[k] = None
                        self.model_dict = model_dict
                    except Exception as e:
                        showerror("Error", "The parameter type of {} is wrong: {}." \
                            .format(k, type(e)))
                
                self.top.destroy()
            def help(self, event=None):
                model_class = str(model)
                start = model_class.find('\'') + 1
                end = model_class.rfind('\'')
                keyword = model_class[start:end]
                keyword = keyword.replace('.', ' ').replace('_', ' ').replace('  ', ' ')
                webbrowser.open('https://google.com/search?q=' + keyword)

        popup = popupWindow(self.master)
        self.master.wait_window(popup.top)
        try:
            self.model_dict = popup.model_dict
        except AttributeError:
            pass

    def setMetricType(self, metric_index):
        def _setMetricType():
            if len(self.output_cols) == 0:
                showerror("Error", "No output variable selected!")
            if self.output_cols[0] in self.numeric_cols:
                if metric_index >= len(self.reg_metrics): # regression
                    for j in range(len(self.metric_selection_bool)):
                        self.metric_selection_bool[j].set(value=False)
                    return showerror("Error", "This metric is for classification!")
            elif self.output_cols[0] in self.categorical_cols:
                if metric_index < len(self.reg_metrics):
                    for j in range(len(self.metric_selection_bool)):
                        self.metric_selection_bool[j].set(value=False)
                    return showerror("Error", "This metric is for regression!")

            for j in range(len(self.metric_selection_bool)):
                self.metric_selection_bool[j].set(value=False)
            self.metric_selection_bool[metric_index].set(value=True)

            # get selected metric type
            for i, b in enumerate(self.metric_selection_bool):
                if b.get():
                    self.metric_int = i

            # set name and metric itself
            if self.output_cols[0] in self.numeric_cols:
                self.metric_name = list(self.reg_metrics.keys())[self.metric_int]
                self.metric = self.reg_metrics[self.metric_name]
            elif self.output_cols[0] in self.categorical_cols:
                metric_int = self.metric_int - len(self.reg_metrics)
                self.metric_name = list(self.clas_metrics.keys())[metric_int]
                self.metric = self.clas_metrics[self.metric_name]

        return _setMetricType

    def show_data(self):
        """
        Opens a new window showing the data.
        """
        data = self.data

        class DataFrame(Frame):
            def __init__(self, master=None):
                top = self.top = Toplevel(master)
                top.attributes("-topmost", True)
                top.geometry('600x720')
                top.title('Data')
                self.table = pt = Table(top, dataframe=data)
                pt.show()
                return

        popup = DataFrame(self.master)
        self.master.wait_window(popup.top)

    def show_description(self):
        """
        Opens a new window showing the a description of the data.
        """
        data = self.data.describe()
        data.reset_index(level=0, inplace=True)

        class DataFrame(Frame):
            def __init__(self, master=None):
                top = self.top = Toplevel(master)
                top.attributes("-topmost", True)
                top.geometry('550x220')
                top.title('Data Description')
                self.table = pt = Table(top, dataframe=data)
                pt.show()
                return

        popup = DataFrame(self.master)
        self.master.wait_window(popup.top)

    def pair_plot(self, limit: int=50):
        """
        Draws a seaborn pairplot of the current data selection.

        Arguments
        ---------
            limit: int, maximum number of data points to plot.

        Returns
        -------
            -
        """
        if len(self.input_cols) + len(self.output_cols) <= 1:
            return showerror("Error", "No features selected to plot.")

        if len(self.output_cols) == 1:
            hue = self.cols[self.output_cols[0]] if \
                    self.bools[self.output_cols[0]][4].get() else None
        else:
            hue = None

        if limit < self.data.shape[0]:
            showinfo("Info", "Showing only the first {} entries.".format(limit))
        
        sns.pairplot(
                self.data[self.cols[self.input_cols + self.output_cols]][:limit],
                hue = hue,
                height = 1.5,
                aspect = 1.
            )
        plt.gcf().canvas.set_window_title('Pair Plot of Data')
        plt.get_current_fig_manager().window.setWindowIcon(
                QtGui.QIcon(os.path.join(os.path.dirname(__file__), 'icon.ico')))
        plt.show()

    def shuffle_data(self):
        """
        Shuffle the rows of the data.
        """
        self.data = self.data.sample(frac=1)

    def startModel(self):
        """
        Main action for trainig the selected model on the data.
        """
        error_msg = ""
        if len(self.output_cols) != 1:
            error_msg += "  * There must be exactly one dependent variable.\n"
        if len(self.input_cols) < 1:
            error_msg += "  * There must be at least one independent variable.\n"
        if not hasattr(self, 'model'):
            error_msg += "  * A model has to be selected.\n"
        if not hasattr(self, 'metric_int'):
            error_msg += "  * A metric has to be selected.\n"
        if len(error_msg) > 0:
            return showerror("Error", 'Model training failed!\n' + error_msg)

        # split data
        X_train, X_test, y_train, y_test = train_test_split(
            self.data[self.cols[self.input_cols]],
            self.data[self.cols[self.output_cols]],
            test_size = self.train_test_ratio)

        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())])
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))])
        preprocessor = ColumnTransformer(transformers=[
            ('num', numeric_transformer, self.cols[
                list(set(self.numeric_cols) - set(self.ignore_cols + self.output_cols))]),
            ('cat', categorical_transformer, self.cols[
                list(set(self.categorical_cols) - set(self.ignore_cols + self.output_cols))])
        ])
        
        model = Pipeline(steps=[('preprocessor', preprocessor),
            ('model', Model(self.master, self.model, self.model_name, self.model_dict))])

        try:
            model.fit(X_train, y_train)
        except ValueError as e:
            return showerror("Error", "Could not train the model (ValueError): {}".format(e))
        except Exception as e:
            return showerror("Error", "Could not train the model: {}.".format(e))

        # make prediction
        try:
            y_pred = model.predict(X_test)
            self.data["predictions"] = model.predict(self.data[self.cols[self.input_cols]])
        except Exception as e:
            return showerror("Error", "Failed to calculate predictions: {}".format(e))

        # evaluate model
        try:
            self.score = np.round(self.metric(y_test, y_pred), 4)
            showinfo("Result", "The model {} scored a {} of {} on the test data." \
                .format(self.model_name, self.metric_name, self.score))
            self.score_str.set("The current model scores {} on the test data.".format(
                    str(np.round(self.score, 4))) if self.score != -1 else "")
        except Exception as e:
            return showerror("Error", "Failed to evaluate the predictions with " \
                + "the specified metric: {}.".format(e))

    def export_data(self):
        """
        Save preprocessed data and predictions (if made).
        """
        try:
            export_file_path = filedialog.asksaveasfilename(
                filetypes = [("default", "*.csv")],
                defaultextension = '.csv'
            )
            self.data.to_csv(export_file_path, index=False, header=True)
        except FileNotFoundError:
            pass
        except Exception as e:
            showerror("Error", "Data could not be saved: {}.".format(e))

    def export_model(self):
        """
        Save the model on disk.
        """
        try:
            export_file_path = filedialog.asksaveasfilename(
                filetypes = [("default", "*.pkl")],
                defaultextension = '.pkl'
            )
            saving = (self.model_name, self.model_dict, self.model,
                      self.model_int, self.metric_int)
            pickle.dump(saving, open(export_file_path, "wb"))
        except (AttributeError, FileNotFoundError):
            pass
        except Exception as e:
            showerror("Error", "Failed to save model: {}.".format(e))
        
    def laod_model(self):
        try:
            filename = filedialog.askopenfilename(
                filetypes = [("default", "*.pkl")],
                defaultextension = '.pkl'
            )
            self.model_name, self.model_dict, self.model, self.model_int, \
                self.metric_int = pickle.load(open(filename, 'rb'))
            
            self.model_loaded = True
            self.setModelType(self.model_int, default_params=False)()
            self.setMetricType(self.metric_int)()
        except FileNotFoundError:
            pass
        except Exception as e:
            showerror("Error", "Data could not be saved: {}.".format(e))

class Model():
    def __init__(
            self, master, model_type,
            model_name: str="", params: dict={}
        ):
        self.master = master
        self.model_type = model_type
        self.model_name = model_name
        self.params = params
        self.model = model_type(**params)

    def fit(self, X, y):
        self.model.fit(X, y)
        load = LoadingWindow(self.master, self.model_name)
        for _ in range(100):
            load.progress() # fake
        load.progress()
        load.destroy()
        return load

    def score(self, X, y):
        return self.model.score(X, y)

    def predict(self, X):
        return self.model.predict(X)

class LoadingWindow():
    def __init__(self, master, model_name):
        self.window = Toplevel()
        self.model_name = model_name
        self.window.title("Training {} Model".format(self.model_name))
        self.window.attributes("-topmost", True)
        self.window.geometry("500x75")
        self.window.grid()
        self.window.resizable(0, 0)
        Label(self.window, text="").grid(row=0, column=0, sticky='N')
        Label(self.window, text="        Training Progress  ") \
            .grid(row=1, column=0, sticky='N')
        self.progress_var = DoubleVar(0)
        self.progressbar = Progressbar(self.window, variable=self.progress_var,
                                       orient="horizontal", length=300,
                                       mode="determinate",  maximum=100)
        self.progressbar.grid(row=1, column=1, sticky='N')
        self.window.pack_slaves()

    def progress(self):
        if self.progressbar["value"] < 100:
            time.sleep(.01)
            self.window.update()
            self.progress_var.set(self.progress_var.get() + 1)
    
    def destroy(self):
        self.window.update()
        self.window.title("Training {} Model: Complete!".format(self.model_name))
        time.sleep(1)
        self.window.destroy()



if __name__ == "__main__":

    root = Tk()
    root.geometry("475x565")
    root.iconbitmap(default=os.path.join(os.path.dirname(__file__), 'icon.ico'))
    root.resizable(0, 0) # Don't allow resizing in the x or y direction
    try: ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(" ")
    except: pass # set taskbar icon (only Windows support?)

    app = MainWindow(root)
    app.mainloop()
