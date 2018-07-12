# -*- coding: utf-8 -*-
"""
Created on Fri Jun  1 20:22:21 2018

@author: XMeng
"""

def map_matrix():
    chessboardTemp = [[[0 for i in range(10)]for j in range(10)]for k in range(10)]
    bulidings=[{ "x": 1, "y": 2, "l": 3, "w": 2, "h": 3 },
               { "x": 5, "y": 2, "l": 2, "w": 3, "h": 5 }]
    #print(bulidings)
    for item in bulidings:
        for x in range(item["x"],item["x"]+item["l"]):
            for y in range(item["y"],item["y"]+item["w"]):
                for z in range(0,item['h']):
                    chessboardTemp[x][y][z] = 1
                
    [x,y,z,l,w,h]=[item["x"],item["y"],0,item["l"],item["w"],item["h"]]
    point=[]
    for item in bulidings:
        point.append([x-1,y-1,z])
        point.append([x-1,y-1,z+h])
        point.append([x-1,y+w,z])
        point.append([x-1,y+w,z+h])
        point.append([x+l,y-1,z])
        point.append([x+l,y-1,z+h])
        point.append([x+l,y+w,z])
        point.append([x+l,y+w,z+h])
        print(chessboardTemp)