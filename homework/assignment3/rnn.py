#!/usr/bin/env python3

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import math
import time
import itertools
import shutil
import tensorflow as tf
import tree as tr
from utils import Vocab
import six

RESET_AFTER = 20
class Config(object):
    """Holds model hyperparams and data information.
       Model objects are passed a Config() object at instantiation.
    """
    embed_size = 15
    label_size = 2
    early_stopping = 2
    anneal_threshold = 0.99
    anneal_by = 1.5
    max_epochs = 30
    lr = 1e-2
    l2 = 2e-3
    model_name = 'rnn_embed=%d_l2=%f_lr=%f.weights'%(embed_size, l2, lr)


class RNN_Model():

    def load_data(self):
        """Loads train/dev/test data and builds vocabulary."""
        self.train_data, self.dev_data, self.test_data = tr.simplified_data(2000, 500, 500)
        #self.train_data, self.dev_data, self.test_data = tr.simplified_data(100, 500, 500)
        # build vocab from training data
        self.vocab = Vocab()
        train_sents = [t.get_words() for t in self.train_data]
        self.vocab.construct(list(itertools.chain.from_iterable(train_sents)))

    def inference(self, tree, predict_only_root=False):
        """For a given tree build the RNN models computation graph up to where it
            may be used for inference.
        Args:
            tree: a Tree object on which to build the computation graph for the RNN
        Returns:
            softmax_linear: Output tensor with the computed logits.
        """
        node_tensors = self.add_model(tree.root)
        if predict_only_root:
            node_tensors = node_tensors[tree.root]
        else:
            node_tensors = [tensor for node, tensor in six.iteritems(node_tensors) if node.label!=2]
            node_tensors = tf.concat(node_tensors,axis=0)
        return self.add_projections(node_tensors)

    def add_model_vars(self):
        '''
        You model contains the following parameters:
            embedding:  tensor(vocab_size, embed_size)
            W1:         tensor(2* embed_size, embed_size)
            b1:         tensor(1, embed_size)
            U:          tensor(embed_size, output_size)
            bs:         tensor(1, output_size)
        Hint: Add the tensorflow variables to the graph here and *reuse* them while building
                the compution graphs for composition and projection for each tree
        Hint: Use a variable_scope "Composition" for the composition layer, and
              "Projection") for the linear transformations preceding the softmax.
        '''
        with tf.variable_scope('Composition',reuse=tf.AUTO_REUSE):
            self.embedding = tf.get_variable("L", dtype = tf.float32,
                                             initializer = tf.random_uniform(shape=(len(self.vocab),
                                                                                    self.config.embed_size),
                                                                             minval = -1.0/np.sqrt(self.config.embed_size),
          maxval = 1.0/np.sqrt(self.config.embed_size)))                                                                   
            self.W1 = tf.get_variable("W1", dtype = tf.float32,
                                           initializer = tf.random_uniform(shape = (2*self.config.embed_size,
                                                                                    self.config.embed_size),
                                                                           minval = -1.0/np.sqrt(self.config.embed_size),
                                                                           maxval = 1.0/np.sqrt(self.config.embed_size)))
            self.b1 = tf.get_variable("b1", dtype = tf.float32, shape = (1,self.config.embed_size),
                                      initializer=tf.zeros_initializer())

            ### END YOUR CODE
        with tf.variable_scope('Projection',reuse=tf.AUTO_REUSE):
            ### YOUR CODE HERE
            self.U = tf.get_variable("U", dtype=tf.float32,
                                     initializer=tf.random_uniform(shape=(self.config.embed_size,self.config.label_size),
                                                                   minval=-1.0/np.sqrt(self.config.embed_size),
                                                                                       maxval=1.0/np.sqrt(self.config.embed_size)))
            self.bs = tf.get_variable("b2", dtype=tf.float32, shape = (1,self.config.label_size),
                                      initializer=tf.zeros_initializer())
            ### END YOUR CODE

    def add_model(self, node):
        """Recursively build the model to compute the phrase embeddings in the tree

        Hint: Refer to tree.py and vocab.py before you start. Refer to
              the model's vocab with self.vocab
        Hint: Reuse the "Composition" variable_scope here
        Hint: Store a node's vector representation in node.tensor so it can be
              used by it's parent
        Hint: If node is a leaf node, it's vector representation is just that of the
              word vector (see tf.gather()).
        Args:
            node: a Node object
        Returns:
            node_tensors: Dict: key = Node, value = tensor(1, embed_size)
        """
        with tf.variable_scope('Composition', reuse=tf.AUTO_REUSE):
            # W1 = tf.get_variable("W1", dtype = tf.float32,
            #                                initializer = tf.random_uniform(shape = (2*self.config.embed_size,
            #                                                                         self.config.embed_size),
            #                                                                minval = -1.0,
            #                                                                maxval = 1.0))
            # b1 = tf.get_variable("b1", dtype = tf.float32, shape = (1,self.config.embed_size),
            #                           initializer=tf.zeros_initializer())

            node_tensors = dict()
            curr_node_tensor = None
            if node.isLeaf:
            ### YOUR CODE HERE
                curr_node_tensor = tf.reshape(tf.nn.embedding_lookup(self.embedding,self.vocab.encode(node.word)),
                                              shape=(1,-1))
                node.tensor = curr_node_tensor
            ### END YOUR CODE
            else:
                node_tensors.update(self.add_model(node.left))
                node_tensors.update(self.add_model(node.right))
                curr_node_tensor = tf.nn.relu(tf.matmul(tf.concat(values=[node.left.tensor,
                                                                   node.right.tensor],axis=1),self.W1)+self.b1)
                node.tensor = curr_node_tensor
            ### END YOUR CODE
        node_tensors[node] = curr_node_tensor
        return node_tensors

    def add_projections(self, node_tensors):
        """Add projections to the composition vectors to compute the raw sentiment scores

        Hint: Reuse the "Projection" variable_scope here
        Args:
            node_tensors: tensor(?, embed_size)
        Returns:
            output: tensor(?, label_size)
        """
        logits = None
        ### YOUR CODE HERE
        with tf.variable_scope('Projection', reuse=tf.AUTO_REUSE):
            logits = tf.matmul(node_tensors,self.U)+self.bs
        ### END YOUR CODE
        return logits

    def loss(self, logits, labels):
        """Adds loss ops to the computational graph.

        Hint: Use sparse_softmax_cross_entropy_with_logits
        Hint: Remember to add l2_loss (see tf.nn.l2_loss)
        Args:
            logits: tensor(num_nodes, output_size)
            labels: python list, len = num_node
        Returns:
            loss: tensor 0-D
        """
        with tf.variable_scope("Projection",reuse=tf.AUTO_REUSE):
            loss = None
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels,logits=logits)
            loss += self.config.l2*tf.nn.l2_loss(self.U)
            loss += self.config.l2*tf.nn.l2_loss(self.W1)
        # END YOUR CODE
        return loss

    def training(self, loss):
        """Sets up the training Ops.

        Creates an optimizer and applies the gradients to all trainable variables.
        The Op returned by this function is what must be passed to the
        `sess.run()` call to cause the model to train. See

        https://www.tensorflow.org/versions/r0.7/api_docs/python/train.html#Optimizer

        for more information.

        Hint: Use tf.train.GradientDescentOptimizer for this model.
                Calling optimizer.minimize() will return a train_op object.

        Args:
            loss: tensor 0-D
        Returns:
            train_op: tensorflow op for training.
        """
        train_op = None
        # YOUR CODE HERE
        with tf.variable_scope("Projection", reuse=tf.AUTO_REUSE):
            #optimizer = tf.train.AdagradOptimizer(self.config.lr)
            optimizer = tf.train.GradientDescentOptimizer(self.config.lr)
            train_op = optimizer.minimize(loss)
        # END YOUR CODE
        return train_op

    def predictions(self, y):
        """Returns predictions from sparse scores

        Args:
            y: tensor(?, label_size)
        Returns:
            predictions: tensor(?,1)
        """
        predictions = None
        # YOUR CODE HERE
        predictions = tf.argmax(y,axis=1)
        # END YOUR CODE
        return predictions

    def __init__(self, config):
        self.config = config
        self.load_data()

    def predict(self, trees, weights_path, get_loss = False):
        """Make predictions from the provided model."""
        results = []
        losses = []
        for i in range(int(math.ceil(len(trees)/float(RESET_AFTER)))):
            with tf.Graph().as_default(), tf.Session() as sess:
                self.add_model_vars()
                saver = tf.train.Saver()
                saver.restore(sess, weights_path)
                for tree in trees[i*RESET_AFTER: (i+1)*RESET_AFTER]:
                    logits = self.inference(tree, True)
                    predictions = self.predictions(logits)
                    root_prediction = sess.run(predictions)[0]
                    if get_loss:
                        root_label = tree.root.label
                        loss = sess.run(self.loss(logits, [root_label]))
                        losses.append(loss)
                    results.append(root_prediction)
        return results, losses

    def run_epoch(self, new_model = False, verbose=True):
        step = 0
        loss_history = []
        resetted=False
        while step < len(self.train_data):
            with tf.Graph().as_default(), tf.Session() as sess:
                self.add_model_vars()
                for j in range(RESET_AFTER):
                    if j == RESET_AFTER-1:
                        resetted=True
                    if step>=len(self.train_data):
                        break
                    tree = self.train_data[step]
                    logits = self.inference(tree)
                    labels = [l for l in tree.labels if l!=2]
                    loss = self.loss(logits, labels)
                    train_op = self.training(loss)
                    if new_model and step == 0:
                        sess.run(tf.global_variables_initializer())
                    
                    if not new_model and step == 0:
                        saver = tf.train.Saver()
                        saver.restore(sess, './weights/%s.temp'%self.config.model_name)

                    if resetted and j==0:
                        saver = tf.train.Saver()
                        saver.restore(sess, './weights/%s.temp'%self.config.model_name)
                        resetted=False
                            
                    # print('='*32)
                    # print('W1 at step', step)
                    # print(sess.run(self.W1))
                    # print('='*32)

                    loss, _ = sess.run([loss, train_op])
                    loss_history.append(loss)
                    if verbose:                        
                        sys.stdout.write('\r{} / {} :    loss = {}\n'.format(
                            step, len(self.train_data), np.mean(np.concatenate(loss_history))))
                        sys.stdout.flush()
                    step+=1
                saver = tf.train.Saver()
                if not os.path.exists("./weights"):
                    os.makedirs("./weights")
                saver.save(sess, './weights/%s.temp'%self.config.model_name)
        train_preds, _ = self.predict(self.train_data, './weights/%s.temp'%self.config.model_name)
        val_preds, val_losses = self.predict(self.dev_data, './weights/%s.temp'%self.config.model_name, get_loss=True)
        train_labels = [t.root.label for t in self.train_data]
        val_labels = [t.root.label for t in self.dev_data]
        train_acc = np.equal(train_preds, train_labels).mean()
        val_acc = np.equal(val_preds, val_labels).mean()

        print('Training acc (only root node): {}\n'.format(train_acc))
        print('Validation acc (only root node): {}\n'.format(val_acc))
        print(self.make_conf(train_labels, train_preds))
        print(self.make_conf(val_labels, val_preds))
        return train_acc, val_acc, loss_history, np.mean(np.concatenate(val_losses))

    def train(self, verbose=True):
        complete_loss_history = []
        train_acc_history = []
        val_acc_history = []
        prev_epoch_loss = float('inf')
        best_val_loss = float('inf')
        best_val_epoch = 0
        stopped = -1
        for epoch in range(self.config.max_epochs):
            print( 'epoch %d'%epoch)
            if epoch==0:
                train_acc, val_acc, loss_history, val_loss = self.run_epoch(new_model=True)
            else:
                train_acc, val_acc, loss_history, val_loss = self.run_epoch()
            complete_loss_history.extend(loss_history)
            train_acc_history.append(train_acc)
            val_acc_history.append(val_acc)

            #lr annealing
            epoch_loss = np.mean(np.concatenate(loss_history))
            if epoch_loss>prev_epoch_loss*self.config.anneal_threshold:
                self.config.lr/=self.config.anneal_by
                print( 'annealed lr to %f'%self.config.lr)
            prev_epoch_loss = epoch_loss

            #save if model has improved on val
            if val_loss < best_val_loss:
                 #shutil.copyfile('./weights/%s.temp.data-00000-of-00001'%self.config.model_name,
                  #               './weights/%s'%self.config.model_name)
                 best_val_loss = val_loss
                 best_val_epoch = epoch

            # if model has not imprvoved for a while stop
            if epoch - best_val_epoch > self.config.early_stopping:
                stopped = epoch
                #break
        if verbose:
                sys.stdout.write('\r')
                sys.stdout.flush()

        print( '\n\nstopped at %d\n'%stopped)
        print (train_acc_history)
        return {
            'loss_history': np.concatenate(complete_loss_history),
            'train_acc_history': train_acc_history,
            'val_acc_history': val_acc_history
            }

    def make_conf(self, labels, predictions):
        confmat = np.zeros([2, 2])
        for l,p in zip(labels, predictions):
            confmat[l, p] += 1
        return confmat


def test_RNN():
    """Test RNN model implementation.

    You can use this function to test your implementation of the Named Entity
    Recognition network. When debugging, set max_epochs in the Config object to 1
    so you can rapidly iterate.
    """
    config = Config()
    model = RNN_Model(config)
    start_time = time.time()
    stats = model.train(verbose=True)
    print( 'Training time: {}'.format(time.time() - start_time))

    plt.plot(stats['loss_history'])
    plt.title('Loss history')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.savefig("loss_history.png")
    plt.show()

    print( 'Test')
    print( '=-=-=')
    predictions, _ = model.predict(model.test_data, './weights/%s.temp'%model.config.model_name)
    labels = [t.root.label for t in model.test_data]
    test_acc = np.equal(predictions, labels).mean()
    print( 'Test acc: {}'.format(test_acc))

if __name__ == "__main__":
        test_RNN()
