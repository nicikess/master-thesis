from ffcv.loader import Loader, OrderOption
from ffcv.transforms import ToTensor
from ffcv.fields.decoders import NDArrayDecoder, FloatDecoder

class DataloaderUtils():

    def __init__(self, path, batch_size, modalities):
        self.data_loader = Loader(path=path, 
                                batch_size=batch_size,
                                order=OrderOption.RANDOM,
                                num_workers=4,
                                pipelines= {
                                            sentinel_2: [NDArrayDecoder(), Clipping([0, 10_000]), ToTensor()],
                                            #'climate_zone': [FloatDecoder()],
                                            #'elevation_difference_label': [FloatDecoder()],
                                            #'era_5': [NDArrayDecoder()],
                                            #'esa_worldcover': [NDArrayDecoder()],
                                            #'glo_30_dem': [NDArrayDecoder()],
                                            #'multiclass_numeric_label': [NDArrayDecoder()],
                                            #'multiclass_one_hot_label': [NDArrayDecoder()],
                                            #'season_s1': [FloatDecoder()],
                                            #'season_s2': [FloatDecoder()],
                                            #'sentinel_1': [NDArrayDecoder()],
                                            #'sentinel_2': [NDArrayDecoder(), Clipping([0, 10_000]), ToTensor()],
                                }
                            )

    #def 