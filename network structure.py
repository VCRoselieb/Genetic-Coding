neural_dict = { '0' : 
                    {'layer':'keras.layers.Dense',
                 'units' : {
                    '0' : 2,
                    '1' : 4,
                    '2' : 8,
                    '3' : 16,
                    '4' : 32,
                    '5' : 64,
                    '6' : 128
                    },
                'activation' : {
                    '0':'linear',
                    '1':'tanh',
                    '2':'elu',
                    '3':'relu',
                    '4':'sigmoid'
                    }
                },
            '1':
                {'layer':'keras.layers.Conv2D',
                 'filters' : {
                    '0' : 2,
                    '1' : 4,
                    '2' : 8,
                    '3' : 16,
                    '4' : 32,
                    '5' : 64,
                    '6' : 128
                    },
                'activation':{
                    '0':'linear',
                    '1':'tanh',
                    '2':'elu',
                    '3':'relu',
                    '4':'sigmoid'
                    },
                'kernel_size' : {
                    '0': 1,
                    '1': 3
                    },
                'strides' : {
                    '0': 1,
                    '1': 2
                    },
                'padding': {
                    '0':'same'
                    }
                 },
            '2':
                {'layer':'keras.layers.CuDNNGRU',
                 'units' : {
                    '0' : 2,
                    '1' : 4,
                    '2' : 8,
                    '3' : 16,
                    '4' : 32,
                    '5' : 64,
                    '6' : 128
                    },
                'kernel_initializer' : {
                    '0': 'glorot_uniform',
                    '1': 'uniform'
                    }
                },
            '3':
                {'layer':'keras.layers.LSTM',
                 'units': {
                    '0' : 2,
                    '1' : 4,
                    '2' : 8,
                    '3' : 16,
                    '4' : 32,
                    '5' : 64,
                    '6' : 128
                    },
                'activation':{
                    '0':'linear',
                    '1':'tanh',
                    '2':'elu',
                    '3':'relu',
                    '4':'sigmoid'
                    },
                'recurrent_activation':{
                    '0':'hard_sigmoid',
                    '1':'tanh',
                    '2':'elu',
                    '3':'relu',
                    '4':'sigmoid',
                    '5':'linear'
                    },
                'kernel_initializer' : {
                    '0': 'glorot_uniform',
                    '1': 'uniform'
                    }
                }
            }

    def construct_model(sequence,genome):
        """
        Description for Dana and Viviane starts here:
            
            
        This is a network cosntructor method I copied from my own project revolving around neuroevolution
        Above, we have a dictionary that contains all the layers and their parameters that can be used by the genetic algorithm.
        I use two variables here:
            
            sequence is a genome that encodes the sequence of layers. i.e [0,0,0,1,1] for a network with three Dense and 2 Conv2D layers
            
            genome here is a matrix containing all parameters for each layer type that is present in the sequence. For our example above
            this variable would contain 3 lists for the Dense layers and 2 lists for the conv2D layers. i.e. [[[0,0],[5,3],[1,1]],[3,2,1,1,0],[1,4,0,0,0]].
            
            If we look into the neural_dict above, we see that it contains 2 sub-dictionaries for the Dense layer and 5 for the Conv2D layer.
            This makes it easy to add new layers by simply writing the name and the parameters of the new layer into the dictionary.
            Infact, my original program does not even use a dictionary defined in-program but simply reads in a json file so we have everything neatly organized.
            
        When in doubt: always remember that the input this function was originally designed for looks like this:
            sequence = [0,0,0,1,1]
            genome = [[[0,0],[5,3],[1,1]],[3,2,1,1,0],[1,4,0,0,0]].
            
        if you get a ValueError with Input '0' is incompatible with layer conv2d_ expected ndim= m found m-1, you need to add the argument "padding='same'" to the conv2d layer. 
        """
        
        
        
        
        
        
        #To keep track of which gene we are reading from on every chromosome, we build a list to keep track of our indeces. i.e for a dense genome we have a list of 0s for "units" and "activation"
        index_array = np.zeros(len(genome),dtype=int)
        
        #We reserve memory for our model. In the following, we will add layers to our model iteratively
        model = keras.models.Sequential()
        
        
    
        #To add a new layer to the model, we extract the necessary data from the genome and map it to a layer definition provided by the neuron dict
        for index in sequence:
            reader_dict = neural_dict[str(index)]
            #We extract a list containing all parameter names from the dictonary entry relevant for the next layer. So for example ['keras.layers.Dense','units','activation'] for a dense layer
            gene_list = list(reader_dict.keys())
            
            #To compile the entire layer parameters into a keras layer type object, we need to store all values in a string and then evaluate it
            #The first necessary parameters is the layer type name. The name is found in the first entry of the dictonary.
            code = reader_dict[str(gene_list[0])]
            
            gene_list = gene_list[1:]
            #we have added the layer name to the code we want to pass into the compiler below. So we do not need it anymore.
            code += "("
            #We now pass parameters into the layer. Since this is different for every layer, we iterate through our gene_list and add parameter names with the corresponding value from the neural_dict
            gene_index = 0
            for parameter in gene_list:
                code += str(parameter)
                code += "="
                #So for example "units="
                genome_information = genome[index][index_array[index]]
                next_argument = reader_dict[str(parameter)][str(genome_information[gene_index])]
                
                if type(next_argument) is str:
                    #repr is another way of representing strings in python. However, here we do not use strings as something that is put into the console, as is traditionally the case
                    #Instead, strings here are necessary to construct a compilable object. Since we want to extract the value from a variable, we need to pass the variable name as a string.
                    #we need a variable name for some layers because they can store strings themselves.
                    #The eval function below takes every string at face value, meaning that it will not understand what you mean when you pass for example:
                    #"keras.layers.conv2D(filters=32,activation='relu'...)"
                    #it will interpret 'relu' as a variable name and thus will not compile.
                    #If we instead use a variable that points to a specific string, it will work. Thus the repr cast here.
                    argument_copy = repr(next_argument)
                    code+= argument_copy
                else:
                    code += str(next_argument)
                gene_index += 1
                
                #if we have not reached the last entry in our list, we are not done adding parameters to our code and we need to ad a comma.
                if parameter != gene_list[-1]:
                    code += ","
            
           if len(model.layers) == 0:
                code += ',input_shape='+str(input_shape)
                #If this is our first layer, we want to add an input shape argument here so that the first layer can take input from the model.train method.
            code += ")"
            index_array[index] += 1
            
            #The fully constructed string is passed into the online-compiler and subsequentially gets evaluated as a keras layer object. This is then added to the model
            compiled_layer = compile(code,'<string>','eval')
            compiled_layer = eval(compiled_layer)
            model.add(compiled_layer)
    
        #in the end, our model is returned. Note that we have not added our last layer yet, as the construct_model function is universal and therefore indifferent towards the task.
        #If we would want to classify images for example, we would need to add an additional dropout layer and a Dense with the corresponding output dimensionality and a softmax activation function.
        return model