import numpy as np
import warnings

class KMeans(object):
    def __init__(self, data, k=3, init_centers=None, init_method="random"):
        """
        Kmeans cluster
        :param k: the number of cluster classes
        :param data:
        :param init_centers: initial centers of K classes, if set as None, use init_method
        :param init_method: random or advanced if without init_centers input
        """
        assert (init_centers is None) or \
               (isinstance(init_centers, np.ndarray) and init_centers.shape[0] == k)

        self.K = k
        self.data = data
        if init_centers is not None:
            self.init_method = "given"
            self.init_centers = init_centers
        else:
            self.init_method = init_method
            getattr(self, "{}_init".format(init_method))()
        self.iter = 0
        self.centers = [self.init_centers]
        self.classes = np.empty([self.data.shape[0]])

    def random_init(self):
        """
        random initialization
        :return: None
        """
        print("init centers randomly ...")
        min_x, min_y = self.data.min(axis=0)
        max_x, max_y = self.data.max(axis=0)
        C_x = np.random.uniform(min_x, max_x, [self.K, 1])
        C_y = np.random.uniform(min_y, max_y, [self.K, 1])
        self.init_centers = np.concatenate([C_x, C_y], axis=1)

    def advanced_init(self):
        """
        advanced initialization
        :return: None
        """
        print("init centers with the advanced method ...")
        center_indexs = [np.random.randint(self.data.shape[0])]
        selected_centers = [self.data[center_indexs[0]]]
        compute_distance = lambda x, y: np.linalg.norm(x - y)
        while len(center_indexs) < self.K:
            max_d = -999999
            max_index = -1
            for i in range(self.data.shape[0]):
                if i in center_indexs:
                    continue

                min_d = 999999
                for j in center_indexs:
                    d = compute_distance(self.data[j], self.data[i])
                    if d < min_d:
                        min_d = d

                if min_d > max_d:
                    max_d = min_d
                    max_index = i

            center_indexs.append(max_index)
            selected_centers.append(self.data[max_index])

        self.init_centers = np.stack(selected_centers, axis=0)

    def get_current_class(self, p):
        distances = np.sqrt(np.sum((p - self.centers[self.iter]) ** 2, axis=1))
        return np.argmin(distances)

    def step(self):
        for i in range(self.data.shape[0]):
            p = self.data[i].reshape([1, 2])
            c = self.get_current_class(p)
            self.classes[i] = c
        new_centers = self.update_centers()
        self.iter += 1
        self.centers.append(new_centers)

    def update_centers(self):
        new_centers = np.zeros_like(self.centers[-1])
        for i in range(self.K):
            selected_data = self.fetch_data(i) #self.data[np.where(self.classes==i)]
            if selected_data.shape[0] == 0:
                new_centers[i] = self.centers[-1][i]
            else:
                new_centers[i] = selected_data.mean(axis=0)
        return new_centers

    def check_end(self):
        return (self.centers[self.iter] == self.centers[self.iter-1]).all()

    def fetch_data(self, class_index):
        return self.data[np.where(self.classes==class_index)]

    def compute_error(self):
        error = 0
        for i in range(self.K):
            selected_data = self.fetch_data(i)
            selected_center = self.centers[-1][i].reshape([1, 2])
            if selected_data.shape[0] == 0:
                error += 0
            else:
                error += np.mean(np.sum((selected_data - selected_center) ** 2, axis=1))
        error /= self.K
        return error