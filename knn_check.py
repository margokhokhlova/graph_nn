import numpy as np
from sklearn.neighbors import NearestNeighbors

def map_for_dataset(gt, returned_queries, distances = None):
    ''' My implementation for the MAP, when the correct image is always a single image
    input - list of gt labels gt and returned queries - list of lists of returned queries '''
    assert len(gt)==len(returned_queries), "The number of HT indexes is not equal to the number of returned values!"
    av_pr = 0
    images_to_ignore = 0
    for i in range(len(gt)):
        query = np.ravel(returned_queries[i]).tolist()
        gt_query = gt[i]
        try:
            element_rank =query.index(gt_query) + 1  #modify the query according to the gt
        except:
            element_rank = 0


        # if distances is not None:
            # don't count the sample if all the distances are same - probably, it is a white image
            # if abs(distances[i][0][0] - distances[i][0][1] + (distances[i][0][1]-distances[i][0][2])) < 0.000000002:
            #     images_to_ignore +=1
            #     continue
        av_pr += 1/element_rank if element_rank > 0 else 0


    # print("Ignored %d empty images" %images_to_ignore)
    return av_pr/(len(gt))




def test_labels_check(keys_2019, keys_2004):
    ''' returns the confirmed gt labels 2019-2004 (checks the images names) '''
    keys_short_19 = get_keys_as_image_coordinates(keys_2019)
    keys_short_04 = get_keys_as_image_coordinates(keys_2004)
    gt_indexes = []
    for key19 in keys_short_19:
        try:
         gt_indexes.append(keys_short_04.index(key19))

        except ValueError:
            print('cannot find the index %s' % (key19))
            break
    return gt_indexes


def get_keys_as_image_coordinates(keys):
    ''' instead of a full path, return only image coordinates + index'''
    modif_keys = [key[-12:-7] + key[-37:-28] for key in keys]
    # modif_keys = []
    # for key in keys:
    #      modif_keys.append(key[-12:-7] + key[-37:-28])
    return modif_keys


def knn_distance_calculation(query, database, distance = 'cosine', N=5):
    assert(database.shape[0] == query.shape[0]), 'The database and query size is not equal in Knn dist calc file'
    num_img = len(database)
    gt_indexes = np.arange(query.shape[0])
    # knn
    knn_array = []
    dist_array = []
    # fit the K-nn algo
    neigh = NearestNeighbors(n_neighbors=N, algorithm='brute', metric=distance)
    neigh.fit(database)
    for i in range(num_img):
        dist, indexes = neigh.kneighbors([query[i, :]])
        knn_array.append(indexes)  # workaround for structure
        dist_array.append(dist)
    print("total of %d graphs were processed" % i)
    map = map_for_dataset(gt_indexes, knn_array, dist_array)
    print("Final MAP for the data is %f" % (map))
    return  map