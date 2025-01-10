# -*- coding: utf-8 -*-
"""
Created on Fri Aug 23 10:38:47 2024

@author: Rodolfo Freitas
"""
import matplotlib.pylab as plt
import matplotlib.colors

colors = [plt.cm.Accent(4),
          plt.cm.tab20c(0),
          plt.cm.tab20c(1), 
          plt.cm.tab10(9),
          plt.cm.tab20c(2),
          plt.cm.tab20c(3),
          #
          plt.cm.tab20c(4),
          plt.cm.tab20c(5),
          plt.cm.tab20c(6),
          plt.cm.tab20c(7),
          #
          plt.cm.tab20b(8),
          plt.cm.tab20b(9),
          plt.cm.tab20b(10),
         
          #
          plt.cm.tab20b(16),
          plt.cm.tab20b(17),
          plt.cm.tab20c(12),
          plt.cm.tab20c(13),
          plt.cm.tab20c(14),
          plt.cm.tab20c(15),
           #
          plt.cm.tab20b(4),
          plt.cm.tab20b(5),
          plt.cm.tab20c(8),
          plt.cm.tab20c(9),
          plt.cm.tab20b(6),
          plt.cm.tab20c(10),
          plt.cm.tab20c(11)]

 
my_cmap=matplotlib.colors.ListedColormap(colors)