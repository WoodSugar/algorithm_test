# -*- coding: utf-8 -*-
"""
@Time   : 2020/8/16

@Author : Shen Fang
"""
import csv
import h5py
import numpy as np
from sklearn.preprocessing import OneHotEncoder


class Process:
    def __init__(self,  ori_flow_file, ori_graph_file, **kwargs):
        self.ori_flow_file = ori_flow_file
        self.ori_graph_file = ori_graph_file

        self.poi_file = kwargs["ori_poi_file"] if "ori_poi_file" in kwargs else None
        self.wea_file = kwargs["ori_wea_file"] if "ori_wea_file" in kwargs else None
        self.time_file = kwargs["ori_time_file"] if "ori_time_file" in kwargs else None

    def get_flow_data(self):
        """
        :return: flow_data, [N, T, C].
        """
        file_obj = h5py.File(self.ori_flow_file, "r")
        flow_data = file_obj["data"][:]

        return flow_data

    def get_graph_data(self):
        """
        :return: the graph structure data, [N, N].
        """
        graph = []
        with open(self.ori_graph_file, "r") as fr:
            reader = csv.reader(fr)
            for item in reader:
                graph.append([float(i) for i in item])

        return np.array(graph)

    def get_poi_data(self):
        """
        :return: [N, C]
        """
        file_obj = h5py.File(self.poi_file, "r")
        poi_data = file_obj["num_pois"][:]  # [N, K]

        total_poi = np.sum(poi_data, axis=1, keepdims=True)  # [N, 1]

        return np.concatenate([poi_data, total_poi], axis=1)  # [N, K + 1]

    def get_weather_data(self):
        """
        :return: [T, C]
        """
        file_obj = h5py.File(self.wea_file, "r")
        weather_data = file_obj["weather_data"][:]

        one_hot_encoder = OneHotEncoder(categorical_features=[-1])
        weather_data = one_hot_encoder.fit_transform(weather_data).toarray()

        if weather_data.shape[1] == 14:
            weather_data = np.delete(weather_data, [5, 8], axis=1)
        return weather_data

    def get_time_data(self):
        """
        :return: [T, C]
        """
        file_obj = h5py.File(self.time_file, "r")
        time_data = file_obj["type_data"][:]

        return time_data

    def save_flow_data(self, flow_file):
        flow_data = self.get_flow_data()
        np.save(flow_file, flow_data)

    def save_graph_data(self, graph_file):
        graph_data = self.get_graph_data()
        np.save(graph_file, graph_data)

    def save_poi_data(self, poi_file):
        poi_data = self.get_poi_data()
        np.save(poi_file, poi_data)

    def save_weather_data(self, wea_file):
        wea_data = self.get_weather_data()
        np.save(wea_file, wea_data)

    def save_time_data(self, time_file):
        time_data = self.get_time_data()
        np.save(time_file, time_data)

    def final_ready_file(self, **kwargs):
        if "flow_file" in kwargs:
            self.save_flow_data(kwargs["flow_file"])

        if "graph_file" in kwargs:
            self.save_graph_data(kwargs["graph_file"])

        if "poi_file" in kwargs:
            self.save_poi_data(kwargs["poi_file"])

        if "wea_file" in kwargs:
            self.save_weather_data(kwargs["wea_file"])

        if "time_file" in kwargs:
            self.save_time_data(kwargs["time_file"])

        return


if __name__ == '__main__':
    # subway = Process(ori_flow_file="Beijing_Subway/Flow_data.h5",
    #                  ori_graph_file="Beijing_Subway/Graph_data.csv",
    #                  ori_poi_file="Beijing_Subway/POI_data.h5",
    #                  ori_wea_file="Beijing_Subway/Weather_data.h5",
    #                  ori_time_file="Beijing_Subway/Time_type.h5")
    #
    # subway.final_ready_file(flow_file="Ready_Subway/flow.npy",
    #                         graph_file="Ready_Subway/graph.npy",
    #                         poi_file="Ready_Subway/poi.npy",
    #                         wea_file="Ready_Subway/wea.npy",
    #                         time_file="Ready_Subway/time.npy")

    # taxi = Process(ori_flow_file="Beijing_Taxi/Flow_data.h5",
    #                  ori_graph_file="Beijing_Taxi/Graph_data.csv",
    #                  ori_poi_file="Beijing_Taxi/POI_data.h5",
    #                  ori_wea_file="Beijing_Taxi/Weather_data.h5",
    #                  ori_time_file="Beijing_Taxi/Time_type.h5")
    #
    # taxi.final_ready_file(flow_file="Ready_Taxi/flow.npy",
    #                         graph_file="Ready_Taxi/graph.npy",
    #                         poi_file="Ready_Taxi/poi.npy",
    #                         wea_file="Ready_Taxi/wea.npy",
    #                         time_file="Ready_Taxi/time.npy")

    bus = Process(ori_flow_file="Beijing_Bus/Flow_data.h5",
                     ori_graph_file="Beijing_Bus/Graph_data.csv",
                     ori_poi_file="Beijing_Bus/POI_data.h5",
                     ori_wea_file="Beijing_Bus/Weather_data.h5",
                     ori_time_file="Beijing_Bus/Time_type.h5")

    bus.final_ready_file(flow_file="Ready_Bus/flow.npy",
                            graph_file="Ready_Bus/graph.npy",
                            poi_file="Ready_Bus/poi.npy",
                            wea_file="Ready_Bus/wea.npy",
                            time_file="Ready_Bus/time.npy")
