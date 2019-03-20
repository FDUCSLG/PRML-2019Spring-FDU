import random
import numpy as np
import sys
import utils

#from tensorflow import flags

#FLAGS = flags.FLAGS

def euclideanDistance(v1, v2):
    v1, v2 = np.array(v1, np.float32), np.array(v2, np.float32)
    return np.sqrt(np.sum(np.square(v1 - v2)))
        
class LSH(object):
    def __init__(self):
        hash_table = None
        hash_fns = None
        mix_weights = None
        mod_c = None 
        fn_group_num = None
        fn_each_group_num = None
        r = None 
        data_set = None
        times_record = {}
    
    def count_times(self,index_array):
        print(index_array)
        
        for data_index in index_array:
            if data_index in self.times_record:
                self.times_record[data_index] += 1
            else:
                self.times_record[data_index] = 1
    
    def filter_max(self):
        max_show_time = 0
        max_index_array = []
        print(self.times_record)
        
        for i in self.times_record:
            if self.times_record[i] > max_show_time:
                max_index_array = [i]
                max_show_time = self.times_record[i]
            elif self.times_record[i] == max_show_time:
                max_index_array.append(i)
        return max_show_time, max_index_array
    
    def generate_hash_family(self, v_length, r, hash_fn_num):
        result = []
        for i in range(hash_fn_num):
            a_temp = []
            for j in range(v_length):
                a_temp.append(random.gauss(0,1))
                b_temp = random.uniform(0,r)
            result.append([a_temp, b_temp])
    
        return result

    def generate_hash_vals(self, LSH_hash_fns, v, r):
        hash_vals = []
        
        for hash_fn in LSH_hash_fns:
            hash_val = (np.inner(hash_fn[0], v) + hash_fn[1]) // r 
            hash_vals.append(hash_val)
        return hash_vals 
        

    def combine_hash_vals(self, hash_vals, mix_weights, mod_c):
        return int(sum(np.multiply(hash_vals, mix_weights)))


    def LSH_build(self, data_set, fn_group_num, fn_each_group_num, r):
        # hash_table ...
        hash_table = []
        
        # mix_weight
        mix_weights = [random.randint(-100,100) for i in range(fn_each_group_num)]

        data_dimention = len(data_set[0])
        data_num = len(data_set)
        
        # mod_c is a big prime number
        mod_c = pow(2, 32) - 5
        
        hash_fns = []   


        for i in range(fn_group_num):
            
            hash_table_dir_temp = {}

            LSH_temp = (self.generate_hash_family(data_dimention, r, fn_each_group_num))
            hash_fns.append(LSH_temp)
            
            for data_index in range(data_num):
                # generate k hash values
                hash_vals = self.generate_hash_vals(LSH_temp, data_set[data_index], r)
                
                # combine the group hash_vals into one values using c
                final_value = self.combine_hash_vals(hash_vals, mix_weights, mod_c)
                table_index = final_value 

                if table_index in hash_table_dir_temp:
                    hash_table_dir_temp[table_index].append(data_index)
                else:
                    hash_table_dir_temp[table_index] = [data_index]
                
            hash_table.append(hash_table_dir_temp)

        self.hash_table = hash_table
        self.hash_fns = hash_fns
        self.mix_weights = mix_weights
        self.mod_c = mod_c
        self.fn_group_num = fn_group_num
        self.fn_each_group_num = fn_each_group_num
        self.r = r
        self.data_set = data_set

        return hash_table, hash_fns, mix_weights, mod_c
    
    def query(self, query_data):
        if self.hash_table == None:
            print("you can not do any query before building a complete hash_table")
        else:
            data_index_set = set()
            data_index_set_union = set()
            data_index_set_intersection = set()

            group_query_hash_val = []
            for i in range(self.fn_group_num):
                query_hash_val = self.generate_hash_vals(self.hash_fns[i], query_data, self.r)
                combine_query_hash_val = self.combine_hash_vals(
                                            query_hash_val, self.mix_weights, self.r)
                group_query_hash_val.append(combine_query_hash_val)
            
            self.times_record = {}
            for i, hash_val in enumerate(group_query_hash_val):
                if hash_val in self.hash_table[i]:
                    self.count_times(self.hash_table[i][hash_val])
                    data_index_set_union.update(set(self.hash_table[i][hash_val]))
                    if i == 0:
                        data_index_set_intersection = set(self.hash_table[i][hash_val])
                    else:
                        data_index_set_intersection.intersection_update(
                                                    set(self.hash_table[i][hash_val]))
            max_show_time, max_index_array = self.filter_max()
            print(data_index_set_union)
            print(data_index_set_intersection)
            
            print("max_show time and max_index array:")
            print(max_show_time)
            print(max_index_array)

            min_distance = -1.0
            min_index = -1    

            for data_index in max_index_array:
            #for data_index in data_index_set_union:
            #for data_index in data_index_set_intersection:
                temp_distance = euclideanDistance(query_data, self.data_set[data_index])
                print(temp_distance)
                if min_distance == -1.0:
                    min_distance = temp_distance
                    min_index = data_index
                else:
                    min_index, min_distance = ([data_index, temp_distance] if temp_distance < min_distance else [min_index, min_distance])
                    
            print("LSH anser is :")
            print("min data_index: %d"%(min_index))
            print("min_distance: %f"%(min_distance))
                
            min_distance2 = -1.0
            min_index2 = -1
            for data_index, data_list in enumerate(self.data_set):
                temp_distance2 = euclideanDistance(query_data, data_list)
                if min_distance2 == -1.0:
                    min_distance2 = temp_distance2
                    min_index2 = data_index 
                else:
                    min_index2, min_distance2 = ([data_index, temp_distance2] if temp_distance2 < min_distance2 else [min_index2, min_distance2]) 

            print(min_index2)
            print(min_distance2)

