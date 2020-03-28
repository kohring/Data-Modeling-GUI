
# tkinter imports
from tkinter import *
from tkinter.ttk import *
from tkinter import filedialog, messagebox, simpledialog

# pandastable imports
from pandastable import config, Table, TableModel
from pandastable.headers import ColumnHeader, RowHeader, IndexHeader
from pandastable import images, util, config
from pandastable.dialogs import *

"""
Based on: Farrell, D 2016 DataExplore: An Application for General Data Analysis
in Research and Education. Journal of Open Research Software, 4: e9, DOI:
http://dx.doi.org/10.5334/jors.94.
"""


class CustomColumnHeader(ColumnHeader):
    """
    Custom pandastable.ColumnHeader which has custom options for the columns.
    """
    def __init__(self, parent=None, main_window=None, table=None, bg='gray25'):
        ColumnHeader.__init__(self, parent=parent, table=table, bg=bg)
        self.main_window = main_window

    def handle_right_click(self, event):
        """respond to a right click"""
        if self.table.enable_menus == False:
            return
        self.rightmenu = self.popupMenu(event)
        return

    def handle_left_click(self, event):
        self.handle_right_click(event)

    def popupMenu(self, event, rows=None, cols=None, outside=None):
        """Add left and right click behaviour for canvas, should not have to override
            this function, it will take its values from defined dicts in constructor"""

        col = self.table.get_col_clicked(event)

        popupmenu = Menu(self, tearoff=0)
        def popupFocusOut(event):
            popupmenu.unpost()

        popupmenu.add_checkbutton(label="Input", command=self.main_window.setIn(col),
                                  variable=self.main_window.bools[col][0])
        popupmenu.add_checkbutton(label="Output", command=self.main_window.setOut(col),
                                  variable=self.main_window.bools[col][1])
        popupmenu.add_checkbutton(label="Ignore", command=self.main_window.setIgnore(col),
                                  variable=self.main_window.bools[col][2])
        popupmenu.add_separator()
        popupmenu.add_checkbutton(label="Numeric", command=self.main_window.setNum(col),
                                  variable=self.main_window.bools[col][3])
        popupmenu.add_checkbutton(label="Categorical", command=self.main_window.setCat(col),
                                  variable=self.main_window.bools[col][4])

        popupmenu.bind("<FocusOut>", popupFocusOut)
        popupmenu.focus_set()
        popupmenu.post(event.x_root, event.y_root)
        applyStyle(popupmenu)
        return popupmenu


class CustomRowHeader(RowHeader):
    """
    Custom pandastable.RowHeader which doesn't allow any right-click changes.
    """
    def __init__(self, parent=None, table=None, width=50):
        RowHeader.__init__(self, parent=parent, table=table, width=width)

    def handle_right_click(self, event):
        return


class CustomTable(Table):
    """
    Custom pandastable.Table which uses CustomRowHeader and CustomColumnHeader.
    """
    def __init__(self, parent=None, main_window=None, model=None, dataframe=None, width=None,
                 height=None, rows=20, cols=5, showtoolbar=False, showstatusbar=False,
                 editable=False, enable_menus=True, **kwargs):
        Table.__init__(self, parent=parent, model=model, dataframe=dataframe,
                       width=width, height=height, rows=rows, cols=cols,
                       showtoolbar=showtoolbar, showstatusbar=showstatusbar,
                       editable=editable, enable_menus=enable_menus,
                       **kwargs)
        self.main_window = main_window

    def show(self, callback=None):
        """Adds column header and scrollbars and combines them with
           the current table adding all to the master frame provided in constructor.
           Table is then redrawn."""

        #Add the table and header to the frame
        self.rowheader = CustomRowHeader(self.parentframe, self)
        self.tablecolheader = CustomColumnHeader(self.parentframe, main_window=self.main_window,
                                                 table=self, bg=self.colheadercolor)
        self.rowindexheader = IndexHeader(self.parentframe, self)
        self.Yscrollbar = AutoScrollbar(self.parentframe,orient=VERTICAL,command=self.set_yviews)
        self.Yscrollbar.grid(row=1,column=2,rowspan=1,sticky='news',pady=0,ipady=0)
        self.Xscrollbar = AutoScrollbar(self.parentframe,orient=HORIZONTAL,command=self.set_xviews)
        self.Xscrollbar.grid(row=2,column=1,columnspan=1,sticky='news')
        self['xscrollcommand'] = self.Xscrollbar.set
        self['yscrollcommand'] = self.Yscrollbar.set
        self.tablecolheader['xscrollcommand'] = self.Xscrollbar.set
        self.rowheader['yscrollcommand'] = self.Yscrollbar.set
        self.parentframe.rowconfigure(1,weight=1)
        self.parentframe.columnconfigure(1,weight=1)

        self.rowindexheader.grid(row=0,column=0,rowspan=1,sticky='news')
        self.tablecolheader.grid(row=0,column=1,rowspan=1,sticky='news')
        self.rowheader.grid(row=1,column=0,rowspan=1,sticky='news')
        self.grid(row=1,column=1,rowspan=1,sticky='news',pady=0,ipady=0)

        self.adjustColumnWidths()
        #bind redraw to resize, may trigger redraws when widgets added
        self.parentframe.bind("<Configure>", self.resized) #self.redrawVisible)
        self.tablecolheader.xview("moveto", 0)
        self.xview("moveto", 0)
        if self.showtoolbar == True:
            self.toolbar = ToolBar(self.parentframe, self)
            self.toolbar.grid(row=0,column=3,rowspan=2,sticky='news')
        if self.showstatusbar == True:
            self.statusbar = statusBar(self.parentframe, self)
            self.statusbar.grid(row=3,column=0,columnspan=2,sticky='ew')
        #self.redraw(callback=callback)
        self.currwidth = self.parentframe.winfo_width()
        self.currheight = self.parentframe.winfo_height()
        if hasattr(self, 'pf'):
            self.pf.updateData()
        return

    def popupMenu(self, event, rows=None, cols=None, outside=None):
        """Add left and right click behaviour for canvas, should not have to override
            this function, it will take its values from defined dicts in constructor"""        
        popupmenu = Menu(self, tearoff = 0)
        def popupFocusOut(event):
            popupmenu.unpost()

        popupmenu.bind("<FocusOut>", popupFocusOut)
        popupmenu.focus_set()
        popupmenu.post(event.x_root, event.y_root)
        applyStyle(popupmenu)
        return popupmenu
