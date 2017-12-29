
# coding: utf-8

# In[1]:

#was built with the help of the book "Make your own neural network" by Tariq Rashid 
import numpy
import matplotlib.pyplot
get_ipython().magic('matplotlib inline')

#we need this for the activation function
import scipy.special
#this is so the program  can interact with arrays and do calcuations with them
#this is the library we need for plotting 
import matplotlib.pyplot


class neuralNetwork:
    #iniliazing the neural net 
    def __init__(self,inputnodes,hiddennodes,outputnodes,learningrate):
        #creating and intializing the variables 
        #setting the nodes for the input, hidden and output layer(3 layer neural network)
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes
        #setting up our learning rate 
        self.lr = learningrate
        #creating the array of weights 
        self.wih = numpy.random.normal(0.0,pow(self.hnodes,-0.5),(self.hnodes,self.inodes))
        #now hidden_output layer weights
        self.who = numpy.random.normal(0.0,pow(self.onodes,-0.5),(self.onodes,self.hnodes))
        #the activation function is the sigmoid function 
        self.activation_function = lambda x: scipy.special.expit(x)
        pass
    #function to train the neural network 
    def train(self,inputs_list,targets_list):
        #overall work out the output for a given training example the compare the to desired output and update weights 
        inputs = numpy.array(inputs_list,ndmin=2).T
        targets = numpy.array(targets_list,ndmin=2).T
       
        #calculate the signals into the hidden layer 
        hidden_inputs = numpy.dot(self.wih,inputs)
        #calculate the signals emerging from the hidden layer 
        hidden_outputs = self.activation_function(hidden_inputs)
        
        #calculate signals into final output layer 
        final_inputs = numpy.dot(self.who,hidden_outputs)
        #calculate the signals emerging from final output layers 
        final_outputs = self.activation_function(final_inputs)
        
        #here is where we calclate the output errors 
        output_errors = targets - final_outputs
        #now we must start calculating the errors 
        hidden_errors = numpy.dot(self.who.T, output_errors)
       
        #now we are going to update the weights and links between the hidden and output layers
        self.who += self.lr * numpy.dot((output_errors*final_outputs*(1.0-final_outputs)),numpy.transpose(hidden_outputs))
        #update the weights for the links between the input and hidden layers 
        self.wih += self.lr *numpy.dot((hidden_errors*hidden_outputs*(1.0-hidden_outputs)),numpy.transpose(inputs))
        
        
        pass
    #query the neural network 
    def query(self, input_list):
        #convert inputs from list to 2d array
        inputs = numpy.array(input_list,ndmin=2).T
        
        #calculate now the signals moving into the input layers 
        hidden_inputs = numpy.dot(self.wih,inputs)
        #calculate signals coming from the hidden layer (Just now applying the activation function)
        hidden_outputs = self.activation_function(hidden_inputs)
        
        #calculate the signals coming into the final layer (simply matrix mutiplication with the weights)
        final_inputs = numpy.dot(self.who, hidden_outputs)
        #finally calculate the signals coming from the final 
        final_outputs = self.activation_function(final_inputs)
        #returning the final output 
        return final_outputs
        pass
    #now we are going to create the neural network 
input_nodes = 784
#because 784 numbers per line in our data set
#its a 20 by 28 pixels that make up the image 
hidden_nodes = 100
#10 output nodes because zero to nine strongest signal wins 
output_nodes = 10
#now setting up the learning rate
learning_rate = 0.3
#now creating the instance of then neural network
n = neuralNetwork(input_nodes,hidden_nodes,output_nodes,learning_rate)
#now we are loading the mnist training data file (60,000) 
training_data_file = open("C:\\Users\\Hussein Syed\\Pictures\\Camera Roll\\mnist_train.csv",'r')
training_data_list = training_data_file.readlines()
training_data_file.close()
#going to have to change the data a little bit
#splitting the commas and then get it in the range of our activiation function
#going through the training data set also
#this is also the training stage of the neural network
for record in training_data_list:
    #split the commas 
    all_values = record.split(',')
    #scale the inputs into the range 
    inputs = (numpy.asfarray(all_values[1:])/255.0*0.99)+0.01
    #create the target output values 
    targets = numpy.zeros(output_nodes)+0.01
    #int the data set all first values say the target 
    targets[int(all_values[0])]=0.99
    n.train(inputs, targets)
    pass
#now what we are going to do is put our neural network to the test 
#first lets get the values on the first line
testing_data = open("C:\\Users\\Hussein Syed\\Pictures\\Camera Roll\\mnist_test_10.csv",'r')
testing_data_numbers = testing_data.readlines()
testing_data.close()
allvalues = testing_data_numbers[0].split(',')
#print the label
print(allvalues[0])
#now lets create the image
# this is just to test the first one 
image_array = numpy.asfarray(allvalues[1:]).reshape(28,28)
matplotlib.pyplot.imshow(image_array,cmap='Greys',interpolation='None')
#starting to code to test our neural network 
#going to use the data from the smaller data set to test the neural network just because of load times 
#created to test the score card 
scorecard = []
for record in testing_data_numbers:
    #splitting the commas 
    all_values = record.split(',')
    #in the data set they list the first number as the correct value 
    correct_label = int (all_values[0])
    print (correct_label,"correct_label")
    #now scaling and shifting the inputs 
    inputs = (numpy.asfarray(all_values[1:])/255.0*0.99)
    #querying the network
    outputs = n.query(inputs)
    #now the highest value is what the network decides
    label = numpy.argmax(outputs)
    print(label,"networks answer")
    #now appending to the correct or incorrect list 
    if (label==correct_label):
        #add one to the score card if correct 
        scorecard.append(1)
    else:
            #if wrong add a zero 
            scorecard.append(0)
            pass 
print (scorecard)
#now we can simply calculate the score of the data set 
scorecard_array = numpy.asarray(scorecard)
#getting the score 
score = (scorecard_array.sum()/scorecard_array.size)*100
print("Performance "+str(score))


# In[ ]:




# In[ ]:



