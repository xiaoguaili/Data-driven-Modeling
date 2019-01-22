import tensorflow as tf
import numpy as np
import timeit


class NeuralNetwork:
    # Initialize the class
    def __init__(self, X_u, Y_u, X_f, Y_f, layers):
             
        self.X_u = X_u
        self.Y_u = Y_u
        self.X_f = X_f
        self.Y_f = Y_f
        self.layers = layers

        # Initialize network weights and biases        
        self.weights, self.biases = self.initialize_NN(layers)
        
        # Define Tensorflow session
        self.sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
        
        # Define placeholders and computational graph
        self.X_u_tf = tf.placeholder(tf.float32, shape=(None, self.X_u.shape[1]))
        self.Y_u_tf = tf.placeholder(tf.float32, shape=(None, self.Y_u.shape[1]))
        self.X_f_tf = tf.placeholder(tf.float32, shape=(None, self.X_f.shape[1]))
        self.Y_f_tf = tf.placeholder(tf.float32, shape=(None, self.Y_f.shape[1]))
        
        # Evaluate prediction of Y_u
#        self.U = self.forward_pass(self.X_u_tf)
        self.U = self.net_u(self.X_u_tf)
        
        # Define neural network f
        self.f = self.net_f(self.X_f_tf)
        
        # Evaluate loss
        self.loss = tf.losses.mean_squared_error(self.Y_u_tf, self.U) + tf.losses.mean_squared_error(self.Y_f_tf, self.f)
        
        # Define optimizer        
        self.optimizer = tf.train.AdamOptimizer(1e-3)
        self.train_op = self.optimizer.minimize(self.loss)
        
        # Initialize Tensorflow variables
        init = tf.global_variables_initializer()
        self.sess.run(init)
    
    def net_u(self, X_u):
        return self.forward_pass(X_u)
        
    
    def net_f(self, x):
        u = self.net_u(x)
        u_x = tf.gradients(u, [x])[0]
        u_xx = tf.gradients(u_x, [x])[0]
        f = u_xx - u
        return f
    
    # Initialize network weights and biases using Xavier initialization
    def initialize_NN(self, layers):      
        # Xavier initialization
        def xavier_init(size):
            in_dim = size[0]
            out_dim = size[1]
            xavier_stddev = 1. / np.sqrt((in_dim + out_dim) / 2.)
            return tf.Variable(tf.random_normal([in_dim, out_dim], dtype=tf.float32) * xavier_stddev, dtype=tf.float32)   
        
        weights = []
        biases = []
        num_layers = len(layers) 
        for l in range(0, num_layers-1):
            W = xavier_init(size=[layers[l], layers[l+1]])
            b = tf.Variable(tf.zeros([1,layers[l+1]], dtype=tf.float32), dtype=tf.float32)
            weights.append(W)
            biases.append(b)        
        return weights, biases
       
           
    # Evaluates the forward pass
    def forward_pass(self, H):
        num_layers = len(self.layers)
        for l in range(0,num_layers-2):
            W = self.weights[l]
            b = self.biases[l]
            H = tf.tanh(tf.add(tf.matmul(H, W), b))
        W = self.weights[-1]
        b = self.biases[-1]
        H = tf.add(tf.matmul(H, W), b)
        return H    
    
    
    # Fetches a mini-batch of data
    def fetch_minibatch(self, X, Y, N_batch):
        N = X.shape[0]
        idx = np.random.choice(N, N_batch, replace=False)
        X_batch = X[idx,:]
        Y_batch = Y[idx,:]        
        return X_batch, Y_batch
    
    
    # Trains the model by minimizing the MSE loss
    def train(self, nIter = 10000, batch_size = 100): 

        start_time = timeit.default_timer()
        for it in range(nIter):     
            # Fetch a mini-batch of data
            X_batch, Y_batch = self.fetch_minibatch(self.X_u, self.Y_u, batch_size)
            
            # Define a dictionary for associating placeholders with data
            tf_dict = {self.X_u_tf: X_batch, self.Y_u_tf: Y_batch, self.X_f_tf: self.X_f, self.Y_f_tf: self.Y_f}  
            
            # Run the Tensorflow session to minimize the loss
            self.sess.run(self.train_op, tf_dict)
            
            # Print
            if it % 100 == 0:
                elapsed = timeit.default_timer() - start_time
                loss_value = self.sess.run(self.loss, tf_dict)
                print('It: %d, Loss: %.3e, Time: %.2f' % 
                      (it, loss_value, elapsed))
                start_time = timeit.default_timer()
                
                
    # Evaluates predictions at test points           
    def predict(self, X_star):      
        # Normalize inputs
#        X_star = (X_star - self.Xmean) / self.Xstd
        tf_dict = {self.X_u_tf: X_star}       
        U_pred = self.sess.run(self.U, tf_dict) 
        # De-normalize outputs
#        Y_star = Y_star * self.Ystd + self.Ymean
        return U_pred
    
    def returnF(self, X_f):        
        tf_dict = {self.X_f_tf: X_f}
        F = self.sess.run(self.f, tf_dict)
        return F