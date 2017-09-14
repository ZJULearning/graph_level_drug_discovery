"""Custom Keras Layers.
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

__author__ = "Han Altae-Tran and Bharath Ramsundar"
__copyright__ = "Copyright 2016, Stanford University"
__license__ = "MIT"

import warnings
import numpy as np
import tensorflow as tf
from deepchem.nn import activations
from deepchem.nn import initializations
from deepchem.nn import regularizers
from deepchem.nn import model_ops
from deepchem.nn.copy import Layer
from deepchem.nn.copy import Input
from deepchem.nn.copy import Dense
from deepchem.nn.copy import Dropout


def affine(x, W, b):
  '''
  print(x_shape)
  mini_block = int(4)
  num = 128

  i=0
  x1 = x[:,i*mini_block:(i+1)*mini_block]
  w1 = W[i*mini_block:(i+1)*mini_block,i*mini_block:(i+1)*mini_block]
  y = tf.matmul(x1,w1)
  for i in range(1,int(num / mini_block)):
    x1 = x[:,i*mini_block:(i+1)*mini_block]
    w1 = W[i*mini_block:(i+1)*mini_block,i*mini_block:(i+1)*mini_block]
    y1 = tf.matmul(x1,w1)
    y = tf.concat([y,y1],1)
  y = y + b
  return y
  '''
  #return tf.matmul(x, W*mask) + b
  return tf.matmul(x, W) + b


def tf_affine(x, vm, scope):
  W = vm.var(scope, 'W')
  b = vm.var(scope, 'b')

  return tf.matmul(x, W) + b


def sum_neigh(atoms, deg_adj_lists, max_deg):
  """Store the summed atoms by degree"""
  deg_summed = max_deg * [None]

  # Tensorflow correctly processes empty lists when using concat
  for deg in range(1, max_deg + 1):
    gathered_atoms = tf.gather(atoms, deg_adj_lists[deg - 1])
    # Sum along neighbors as well as self, and store
    summed_atoms = tf.reduce_sum(gathered_atoms, 1)
    deg_summed[deg - 1] = summed_atoms

  return deg_summed


def graph_conv(atoms_in, deg_adj_lists, deg_slice, max_deg, min_deg, W_list,
               b_list, add_time,batch_size,membership):
  """Core tensorflow function implementing graph convolution

  Parameters
  ----------
  atoms: tf.Tensor
    Should be of shape (n_atoms, n_feat)
  deg_adj_lists: list
    Of length (max_deg+1-min_deg). The deg-th element is a list of
    adjacency lists for atoms of degree deg.
  deg_slice: tf.Tensor
    Of shape (max_deg+1-min_deg,2). Explained in GraphTopology.
  max_deg: int
    Maximum degree of atoms in molecules.
  min_deg: int
    Minimum degree of atoms in molecules
  W_list: list
    List of learnable weights for convolution.
  b_list: list
    List of learnable biases for convolution.

  Returns
  -------
  tf.Tensor
    Of shape (n_atoms, n_feat)
  """
  W = iter(W_list)
  b = iter(b_list)
  if add_time == 0:
      atoms = atoms_in
  else:
      atoms = atoms_in[:-batch_size]
      dummy_atoms = atoms_in[-batch_size:]
      
  #Sum all neighbors using adjacency matrix
  deg_summed = sum_neigh(atoms, deg_adj_lists, max_deg)

  # Get collection of modified atom features
  new_rel_atoms_collection = (max_deg + 1 - min_deg) * [None]

  W0, b0 = next(W), next(b)
  for deg in range(1, max_deg + 1):
    # Obtain relevant atoms for this degree
    rel_atoms = deg_summed[deg - 1]

    # Get self atoms
    begin = tf.stack([deg_slice[deg - min_deg, 0], 0])
    size = tf.stack([deg_slice[deg - min_deg, 1], -1])
    self_atoms = tf.slice(atoms, begin, size)

    # Apply hidden affine to relevant atoms and append
    rel_out = affine(rel_atoms, next(W), next(b))
    #self_tmp = affine(self_atoms, next(W), next(b)) 
    #self_out = affine(self_atoms,W0, b0)# next(W), next(b))
    #self_out = affine(self_atoms, next(W), next(b))
    out = rel_out + affine(self_atoms,W0, b0)

    new_rel_atoms_collection[deg - min_deg] = out

  # Determine the min_deg=0 case
  if min_deg == 0:
    deg = 0

    begin = tf.stack([deg_slice[deg - min_deg, 0], 0])
    size = tf.stack([deg_slice[deg - min_deg, 1], -1])
    self_atoms = tf.slice(atoms, begin, size)

    # Only use the self layer
    out = affine(self_atoms, next(W), next(b))

    new_rel_atoms_collection[deg - min_deg] = out

  # Combine all atoms back into the list
  activated_atoms = tf.concat(axis=0, values=new_rel_atoms_collection)
  
  
  atoms_gather = graph_gather(atoms, membership, batch_size)
  
  if 1:
      ##CONV
      W1, b1 = next(W), next(b)
      W2, b2 = next(W), next(b)
      if add_time == 0:
          activated_dummy_atoms = affine(atoms_gather,W1, b1)
      else:
          activated_dummy_atoms = affine(atoms_gather,W1, b1) + affine(dummy_atoms,W2, b2)
  else:
      ## GRU
      W1, b1 = next(W), next(b)
      W2, b2 = next(W), next(b)
      W3, b3 = next(W), next(b)
      W4, b4 = next(W), next(b)
      W5, b5 = next(W), next(b)
      W6, b6 = next(W), next(b)
      if add_time == 0:
          activated_dummy_atoms = affine(atoms_gather,W1, b1)         
      else:
          r = tf.sigmoid(tf.matmul(atoms_gather, W1) + tf.matmul(dummy_atoms, W2))
          z = tf.sigmoid(tf.matmul(atoms_gather, W3) + tf.matmul(dummy_atoms, W4))
          hid = tf.tanh(tf.matmul(atoms_gather, W5) + tf.matmul(r * dummy_atoms, W6))
          activated_dummy_atoms = (1 - z) * dummy_atoms + z * hid
  

  return tf.concat([activated_atoms,activated_dummy_atoms],axis=0)


def graph_gather(atoms, membership_placeholder, batch_size):
  """
  Parameters
  ----------
  atoms: tf.Tensor
    Of shape (n_atoms, n_feat)
  membership_placeholder: tf.Placeholder
    Of shape (n_atoms,). Molecule each atom belongs to.
  batch_size: int
    Batch size for deep model.

  Returns
  -------
  tf.Tensor
    Of shape (batch_size, n_feat)
  """

  # WARNING: Does not work for Batch Size 1! If batch_size = 1, then use reduce_sum!
  assert batch_size > 1, "graph_gather requires batches larger than 1"

  # Obtain the partitions for each of the molecules
  activated_par = tf.dynamic_partition(atoms, membership_placeholder,
                                       batch_size)

  # Sum over atoms for each molecule
  sparse_reps = [
      tf.reduce_sum(activated, 0, keep_dims=True) for activated in activated_par
  ]

  # Get the final sparse representations
  sparse_reps = tf.concat(axis=0, values=sparse_reps)

  return sparse_reps


def graph_pool(atoms, deg_adj_lists, deg_slice, max_deg, min_deg):
  """
  Parameters
  ----------
  atoms: tf.Tensor
    Of shape (n_atoms, n_feat)
  deg_adj_lists: list
    Of length (max_deg+1-min_deg). The deg-th element is a list of
    adjacency lists for atoms of degree deg.
  deg_slice: tf.Tensor
    Of shape (max_deg+1-min_deg,2). Explained in GraphTopology.
  max_deg: int
    Maximum degree of atoms in molecules.
  min_deg: int
    Minimum degree of atoms in molecules

  Returns
  -------
  tf.Tensor
    Of shape (batch_size, n_feat)
  """
  # Store the summed atoms by degree
  deg_maxed = (max_deg + 1 - min_deg) * [None]

  # Tensorflow correctly processes empty lists when using concat

  for deg in range(1, max_deg + 1):
    # Get self atoms
    begin = tf.stack([deg_slice[deg - min_deg, 0], 0])
    size = tf.stack([deg_slice[deg - min_deg, 1], -1])
    self_atoms = tf.slice(atoms, begin, size)

    # Expand dims
    self_atoms = tf.expand_dims(self_atoms, 1)

    # always deg-1 for deg_adj_lists
    gathered_atoms = tf.gather(atoms, deg_adj_lists[deg - 1])
    gathered_atoms = tf.concat(axis=1, values=[self_atoms, gathered_atoms])

    maxed_atoms = tf.reduce_max(gathered_atoms, 1)
    deg_maxed[deg - min_deg] = maxed_atoms

  if min_deg == 0:
    begin = tf.stack([deg_slice[0, 0], 0])
    size = tf.stack([deg_slice[0, 1], -1])
    self_atoms = tf.slice(atoms, begin, size)
    deg_maxed[0] = self_atoms

  return tf.concat(axis=0, values=deg_maxed)

 
class GraphConv(Layer):
  """"Performs a graph convolution.

  Note this layer expects the presence of placeholders defined by GraphTopology
  and expects that they follow the ordering provided by
  GraphTopology.get_input_placeholders().
  """

  def __init__(self,
               nb_filter,
               n_atom_features,
               init='glorot_uniform',
               activation='linear',
               dropout=None,
               max_deg=10,
               min_deg=0,
               **kwargs):
    """
    Parameters
    ----------
    nb_filter: int
      Number of convolutional filters.
    n_atom_features: int
      Number of features listed per atom.
    init: str, optional
      Weight initialization for filters.
    activation: str, optional
      Activation function applied after convolution.
    dropout: float, optional
      Dropout probability.
    max_deg: int, optional
      Maximum degree of atoms in molecules.
    min_deg: int, optional
      Minimum degree of atoms in molecules.
    """
    warnings.warn("The dc.nn.GraphConv is "
                  "deprecated. Will be removed in DeepChem 1.4. "
                  "Will be replaced by dc.models.tensorgraph.layers.GraphConv",
                  DeprecationWarning)
    super(GraphConv, self).__init__(**kwargs)

    self.init = initializations.get(init)  # Set weight initialization
    self.activation = activations.get(activation)  # Get activations
    self.nb_filter = nb_filter  # Save number of filters
    self.dropout = dropout  # Save dropout params
    self.max_deg = max_deg
    self.min_deg = min_deg
    # TODO(rbharath): It's not clear where nb_affine comes from.
    # Is there a solid explanation here?
    self.nb_affine = 2 * max_deg + (1 - min_deg)
    self.n_atom_features = n_atom_features

  def build(self):
    """"Construct internal trainable weights.

    n_atom_features should provide the number of features per atom.

    Parameters
    ----------
    n_atom_features: int
      Number of features provied per atom.
    """
    n_atom_features = self.n_atom_features

    # Generate the nb_affine weights and biases
    self.W_list = [
        self.init([n_atom_features, self.nb_filter])
        for k in range(self.nb_affine)
    ]
    self.b_list = [
        model_ops.zeros(shape=[
            self.nb_filter,
        ]) for k in range(self.nb_affine)
    ]

    self.trainable_weights = self.W_list + self.b_list

  def get_output_shape_for(self, input_shape):
    """Output tensor shape produced by this layer."""
    atom_features_shape = input_shape[0]
    assert len(atom_features_shape) == 2, \
            "MolConv only takes 2 dimensional tensors for x"
    n_atoms = atom_features_shape[0]
    return (n_atoms, self.nb_filter)

  def call(self, x, mask=None):
    """Execute this layer on input tensors.

    This layer is meant to be executed on a Graph. So x is expected to
    be a list of placeholders, with the first placeholder the list of
    atom_features (learned or input) at this level, the second the deg_slice,
    the third the membership, and the remaining the deg_adj_lists.

    Visually

    x = [atom_features, deg_slice, membership, deg_adj_list placeholders...]

    Parameters
    ----------
    x: list
      list of Tensors of form described above.
    mask: bool, optional
      Ignored. Present only to shadow superclass call() method.

    Returns
    -------
    atom_features: tf.Tensor
      Of shape (n_atoms, nb_filter)
    """
    # Add trainable weights
    self.build()

    # Extract atom_features
    atom_features = x[0]

    # Extract graph topology
    deg_slice, membership, deg_adj_lists = x[1], x[2], x[3:-1]
    training = x[-1]

    # Perform the mol conv
    atom_features = graph_conv(atom_features, deg_adj_lists, deg_slice,
                               self.max_deg, self.min_deg, self.W_list,
                               self.b_list)

    atom_features = self.activation(atom_features)

    if self.dropout is not None:
      atom_features = training * tf.nn.dropout(atom_features, self.dropout) + (1 -training) * atom_features
    return atom_features


class GraphResBlock(Layer): 
  """"Performs a graph convolution.

  Note this layer expects the presence of placeholders defined by GraphTopology
  and expects that they follow the ordering provided by
  GraphTopology.get_input_placeholders().
  """

  def __init__(self,
               nb_filter,
               n_atom_features,
               batch_size,
               init='glorot_uniform',
               activation='linear',
               dropout=None,
               max_deg=10,
               min_deg=0,
               epsilon=1e-3,
               mode=0,
               axis=-1,
               momentum=0.99,
               beta_init='zero',
               gamma_init='one',
               gamma_regularizer=None,
               beta_regularizer=None,
               bn_first=False,
               add=True,
               weight_decay =0.0001,
               **kwargs):
    """
    Parameters
    ----------
    nb_filter: int
      Number of convolutional filters.
    n_atom_features: int
      Number of features listed per atom.
    init: str, optional
      Weight initialization for filters.
    activation: str, optional
      Activation function applied after convolution.
    dropout: float, optional
      Dropout probability.
    max_deg: int, optional
      Maximum degree of atoms in molecules.
    min_deg: int, optional
      Minimum degree of atoms in molecules.
    """
	##################GraphConv###################
    warnings.warn("The dc.nn.GraphConv is "
                  "deprecated. Will be removed in DeepChem 1.4. "
                  "Will be replaced by dc.models.tensorgraph.layers.GraphConv",
                  DeprecationWarning)
    super(GraphResBlock, self).__init__(**kwargs)

    self.init = initializations.get(init)  # Set weight initialization
    self.activation = activations.get(activation)  # Get activations
    self.nb_filter = nb_filter  # Save number of filters
    self.dropout = dropout  # Save dropout params
    self.max_deg = max_deg
    self.min_deg = min_deg
    self.weight_decay = weight_decay
    # TODO(rbharath): It's not clear where nb_affine comes from.
    # Is there a solid explanation here?
    self.nb_affine = 2 * max_deg + (1 - min_deg)
    self.n_atom_features = n_atom_features
	
	##################BachNormalization###################
    self.beta_init = initializations.get(beta_init)
    self.gamma_init = initializations.get(gamma_init)
    self.epsilon = epsilon
    self.mode = mode
    self.axis = axis
    self.momentum = momentum
    self.gamma_regularizer = regularizers.get(gamma_regularizer)
    self.beta_regularizer = regularizers.get(beta_regularizer)
    self.bn_first = bn_first
    self.add = add
    self.batch_size = batch_size
    if self.mode == 0:
      self.uses_learning_phase = True

    n = 32
    t=tf.diag(tf.ones([n]))
    t=tf.reshape(t,[n,n,1])
    mask = tf.image.resize_images(t,[128,128],method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    mask = tf.reshape(mask,[128,128])
    self.mask = tf.gather(mask,tf.cast(tf.floormod(tf.range(128)*7,tf.cast(tf.ones([128])*128,tf.int32)),tf.int32))
    #self.mask = tf.random_shuffle(mask)
    #super(BatchNormalization, self).__init__(**kwargs)
    
  def build(self):
    """"Construct internal trainable weights.

    n_atom_features should provide the number of features per atom.

    Parameters
    ----------
    n_atom_features: int
      Number of features provied per atom.
    """
	##################GraphConv###################
    n_atom_features = self.n_atom_features
    # Generate the nb_affine weights and biases

    self.W_list = [
        self.init([n_atom_features, self.nb_filter])# * self.mask
        for k in range(self.nb_affine)
    ]
    self.b_list = [
        model_ops.zeros(shape=[
            self.nb_filter,
        ]) for k in range(self.nb_affine)
    ]
    reg = tf.contrib.layers.l2_regularizer(self.weight_decay)
    for k in range(self.nb_affine):
        tf.add_to_collection('weight_decay',reg(self.W_list[k]))
        tf.add_to_collection('weight_decay',reg(self.b_list[k]))


    self.trainable_weights = self.W_list + self.b_list

	##################BachNormalization###################	
  def build_bn(self, input_shape):
    shape = (input_shape[self.axis],)

    self.gamma = self.add_weight(
        shape,
        initializer=self.gamma_init,
        regularizer=self.gamma_regularizer,
        name='{}_gamma'.format(self.name))
    self.beta = self.add_weight(
        shape,
        initializer=self.beta_init,
        regularizer=self.beta_regularizer,
        name='{}_beta'.format(self.name))
    # Not Trainable
    self.running_mean = self.add_weight(
        shape, initializer='zero', name='{}_running_mean'.format(self.name))
    # Not Trainable
    self.running_std = self.add_weight(
        shape, initializer='one', name='{}_running_std'.format(self.name))
		
  def get_output_shape_for(self, input_shape):
    """Output tensor shape produced by this layer."""
    atom_features_shape = input_shape[0]
    assert len(atom_features_shape) == 2, \
            "MolConv only takes 2 dimensional tensors for x"
    n_atoms = atom_features_shape[0]
    return (n_atoms, self.nb_filter)

  def call(self, x, mask=None):
    """Execute this layer on input tensors.

    This layer is meant to be executed on a Graph. So x is expected to
    be a list of placeholders, with the first placeholder the list of
    atom_features (learned or input) at this level, the second the deg_slice,
    the third the membership, and the remaining the deg_adj_lists.

    Visually

    x = [atom_features, deg_slice, membership, deg_adj_list placeholders...]

    Parameters
    ----------
    x: list
      list of Tensors of form described above.
    mask: bool, optional
      Ignored. Present only to shadow superclass call() method.

    Returns
    -------
    atom_features: tf.Tensor
      Of shape (n_atoms, nb_filter)
    """
	
	##################GraphConv###################
    # Add trainable weights
    self.build()

    # Extract atom_features
    atom_features_ori = x[0]

    # Extract graph topology
    deg_slice, membership, deg_adj_lists = x[1], x[2], x[3:-2]
    training = x[-2]
    add_time = x[-1]

    # Perform the mol conv
    #print(x)
    #print(atom_features_ori)
    #print(deg_adj_lists)
    #print(deg_slice)
    atom_features = graph_conv(atom_features_ori, deg_adj_lists, deg_slice,
                               self.max_deg, self.min_deg, self.W_list,
                               self.b_list,add_time,self.batch_size,membership)
	
    if not self.bn_first:
      atom_features = self.activation(atom_features)
	  
	##################BachNormalization###################
    xx = atom_features
    if not isinstance(xx, list):
      input_shape = model_ops.int_shape(xx)
    else:
      xx = xx[0]
      input_shape = model_ops.int_shape(xx)
    self.build_bn(input_shape)
    if self.mode == 0 or self.mode == 2:

      reduction_axes = list(range(len(input_shape)))
      del reduction_axes[self.axis]
      broadcast_shape = [1] * len(input_shape)
      broadcast_shape[self.axis] = input_shape[self.axis]

      x_normed, mean, std = model_ops.normalize_batch_in_training(
          xx, self.gamma, self.beta, reduction_axes, epsilon=self.epsilon)

      if self.mode == 0:
        self.add_update([
            model_ops.moving_average_update(self.running_mean, mean,
                                            self.momentum),
            model_ops.moving_average_update(self.running_std, std,
                                            self.momentum)
        ], xx)

        if sorted(reduction_axes) == range(model_ops.get_ndim(xx))[:-1]:
          x_normed_running = tf.nn.batch_normalization(
              xx,
              self.running_mean,
              self.running_std,
              self.beta,
              self.gamma,
              epsilon=self.epsilon)
        else:
          # need broadcasting
          broadcast_running_mean = tf.reshape(self.running_mean,
                                              broadcast_shape)
          broadcast_running_std = tf.reshape(self.running_std, broadcast_shape)
          broadcast_beta = tf.reshape(self.beta, broadcast_shape)
          broadcast_gamma = tf.reshape(self.gamma, broadcast_shape)
          x_normed_running = tf.batch_normalization(
              xx,
              broadcast_running_mean,
              broadcast_running_std,
              broadcast_beta,
              broadcast_gamma,
              epsilon=self.epsilon)

        # pick the normalized form of x corresponding to the training phase
        x_normed = model_ops.in_train_phase(x_normed, x_normed_running)

    elif self.mode == 1:
      # sample-wise normalization
      m = model_ops.mean(xx, axis=-1, keepdims=True)
      std = model_ops.sqrt(
          model_ops.var(xx, axis=-1, keepdims=True) + self.epsilon)
      x_normed = (xx - m) / (std + self.epsilon)
      x_normed = self.gamma * x_normed + self.beta
      
    if self.add:
      atom_features = atom_features_ori + x_normed
    else:
      atom_features = x_normed
      
    if self.bn_first:
      atom_features = self.activation(atom_features)

    if self.dropout is not None:
      atom_features = training * tf.nn.dropout(atom_features, self.dropout) + (1 -training) * atom_features
      #atom_features = tf.cond(training, lambda: tf.nn.dropout(atom_features, self.dropout), lambda: atom_features)

    #reg1 = tf.contrib.layers.l1_regularizer(0)
    #tf.add_to_collection('weight_decay',reg1(atom_features))
	  
    return atom_features


class GraphPool(Layer):
  """Performs a pooling operation over an arbitrary graph.

  Performs a max pool over the feature vectors for an atom and its neighbors
  in bond-graph. Returns a tensor of the same size as the input.
  """

  def __init__(self,batch_size, max_deg=10, min_deg=0, **kwargs):
    """
    Parameters
    ----------
    max_deg: int, optional
      Maximum degree of atoms in molecules.
    min_deg: int, optional
      Minimum degree of atoms in molecules.
    """
    warnings.warn("The dc.nn.GraphPool is "
                  "deprecated. Will be removed in DeepChem 1.4. "
                  "Will be replaced by dc.models.tensorgraph.layers.GraphPool",
                  DeprecationWarning)
    self.max_deg = max_deg
    self.min_deg = min_deg
    self.batch_size = batch_size
    super(GraphPool, self).__init__(**kwargs)

  def build(self, input_shape):
    """Nothing needed (no learnable weights)."""
    pass

  def get_output_shape_for(self, input_shape):
    """Output tensor shape produced by this layer."""
    # Extract nodes
    atom_features_shape = input_shape[0]

    assert len(atom_features_shape) == 2, \
            "GraphPool only takes 2 dimensional tensors"
    return atom_features_shape

  def call(self, x, mask=None):
    """Execute this layer on input tensors.

    This layer is meant to be executed on a Graph. So x is expected to
    be a list of placeholders, with the first placeholder the list of
    atom_features (learned or input) at this level, the second the deg_slice,
    the third the membership, and the remaining the deg_adj_lists.

    Visually

    x = [atom_features, deg_slice, membership, deg_adj_list placeholders...]

    Parameters
    ----------
    x: list
      list of Tensors of form described above.
    mask: bool, optional
      Ignored. Present only to shadow superclass call() method.

    Returns
    -------
    tf.Tensor
      Of shape (n_atoms, n_feat), where n_feat is number of atom_features
    """
    # Extract atom_features
    atom_features = x[0]
    atoms = atom_features[:-self.batch_size]
    dummy_atoms = atom_features[-self.batch_size:]

    # Extract graph topology
    deg_slice, membership, deg_adj_lists = x[1], x[2], x[3:-2]

    # Perform the mol gather
    atoms = graph_pool(atoms, deg_adj_lists, deg_slice,
                               self.max_deg, self.min_deg)
    
    
    return tf.concat([atoms,dummy_atoms],axis=0)


	
	
class GraphGather(Layer):
  """Gathers information for each molecule.

  The various graph convolution operations expect as input a tensor
  atom_features of shape (n_atoms, n_feat). However, we train on batches of
  molecules at a time. The GraphTopology object groups a list of molecules
  into the atom_features tensor. The tensorial operations are done on this tensor,
  but at the end, the atoms need to be grouped back into molecules. This
  layer takes care of that operation.

  Note this layer expects the presence of placeholders defined by GraphTopology
  and expects that they follow the ordering provided by
  GraphTopology.get_input_placeholders().
  """

  def __init__(self, batch_size, activation='linear', **kwargs):
    """
    Parameters
    ----------
    batch_size: int
      Number of elements in batch of data.
    """
    warnings.warn(
        "The dc.nn.GraphGather is "
        "deprecated. Will be removed in DeepChem 1.4. "
        "Will be replaced by dc.models.tensorgraph.layers.GraphGather",
        DeprecationWarning)
    super(GraphGather, self).__init__(**kwargs)

    self.activation = activations.get(activation)  # Get activations
    self.batch_size = batch_size

  def build(self, input_shape):
    """Nothing needed (no learnable weights)."""
    pass

  def get_output_shape_for(self, input_shape):
    """Output tensor shape produced by this layer."""
    # Extract nodes and membership
    atom_features_shape = input_shape[0]
    membership_shape = input_shape[2]

    assert len(atom_features_shape) == 2, \
            "GraphGather only takes 2 dimensional tensors"
    n_feat = atom_features_shape[1]

    return (self.batch_size, n_feat)

  def call(self, x, mask=None):
    """Execute this layer on input tensors.

    This layer is meant to be executed on a Graph. So x is expected to
    be a list of placeholders, with the first placeholder the list of
    atom_features (learned or input) at this level, the second the deg_slice,
    the third the membership, and the remaining the deg_adj_lists.

    Visually

    x = [atom_features, deg_slice, membership, deg_adj_list placeholders...]

    Parameters
    ----------
    x: list
      list of Tensors of form described above.
    mask: bool, optional
      Ignored. Present only to shadow superclass call() method.

    Returns
    -------
    tf.Tensor
      Of shape (batch_size, n_feat), where n_feat is number of atom_features
    """
    # Extract atom_features
    atom_features = x[0]
    return self.activation(atom_features[-self.batch_size:])
    # Extract graph topology
    #membership = x[2]

    # Perform the mol gather
    #reg1 = tf.contrib.layers.l1_regularizer(0.0002)
    #tf.add_to_collection('weight_decay',reg1(atom_features))
    #mol_features = graph_gather(atom_features, membership, self.batch_size)
    
    

    #return self.activation(mol_features)


class AttnLSTMEmbedding(Layer):
  """Implements AttnLSTM as in matching networks paper.

  References:
  Matching Networks for One Shot Learning
  https://arxiv.org/pdf/1606.04080v1.pdf

  Order Matters: Sequence to sequence for sets
  https://arxiv.org/abs/1511.06391
  """

  def __init__(self,
               n_test,
               n_support,
               n_feat,
               max_depth,
               init='glorot_uniform',
               activation='linear',
               dropout=None,
               **kwargs):
    """
    Parameters
    ----------
    n_support: int
      Size of support set.
    n_test: int
      Size of test set.
    n_feat: int
      Number of features per atom
    max_depth: int
      Number of "processing steps" used by sequence-to-sequence for sets model.
    init: str, optional
      Type of initialization of weights
    activation: str, optional
      Activation for layers.
    dropout: float, optional
      Dropout probability
    """
    super(AttnLSTMEmbedding, self).__init__(**kwargs)

    self.init = initializations.get(init)  # Set weight initialization
    self.activation = activations.get(activation)  # Get activations
    self.max_depth = max_depth
    self.n_test = n_test
    self.n_support = n_support
    self.n_feat = n_feat

  def get_output_shape_for(self, input_shape):
    """Returns the output shape. Same as input_shape.

    Parameters
    ----------
    input_shape: list
      Will be of form [(n_test, n_feat), (n_support, n_feat)]

    Returns
    -------
    list
      Of same shape as input [(n_test, n_feat), (n_support, n_feat)]
    """
    x_input_shape, xp_input_shape = input_shape  #Unpack

    return input_shape

  def call(self, x_xp, mask=None):
    """Execute this layer on input tensors.

    Parameters
    ----------
    x_xp: list
      List of two tensors (X, Xp). X should be of shape (n_test, n_feat) and
      Xp should be of shape (n_support, n_feat) where n_test is the size of
      the test set, n_support that of the support set, and n_feat is the number
      of per-atom features.

    Returns
    -------
    list
      Returns two tensors of same shape as input. Namely the output shape will
      be [(n_test, n_feat), (n_support, n_feat)]
    """
    # x is test set, xp is support set.
    x, xp = x_xp

    ## Initializes trainable weights.
    n_feat = self.n_feat

    self.lstm = LSTMStep(n_feat, 2 * n_feat)
    self.q_init = model_ops.zeros([self.n_test, n_feat])
    self.r_init = model_ops.zeros([self.n_test, n_feat])
    self.states_init = self.lstm.get_initial_states([self.n_test, n_feat])

    self.trainable_weights = [self.q_init, self.r_init]

    ### Performs computations

    # Get initializations
    q = self.q_init
    #r = self.r_init
    states = self.states_init

    for d in range(self.max_depth):
      # Process using attention
      # Eqn (4), appendix A.1 of Matching Networks paper
      e = cos(x + q, xp)
      a = tf.nn.softmax(e)
      r = model_ops.dot(a, xp)

      # Generate new aattention states
      y = model_ops.concatenate([q, r], axis=1)
      q, states = self.lstm([y] + states)  #+ self.lstm.get_constants(x)

    return [x + q, xp]

  def compute_mask(self, x, mask=None):
    if not (mask is None):
      return mask
    return [None, None]


class ResiLSTMEmbedding(Layer):
  """Embeds its inputs using an LSTM layer."""

  def __init__(self,
               n_test,
               n_support,
               n_feat,
               max_depth,
               init='glorot_uniform',
               activation='linear',
               **kwargs):
    """
    Unlike the AttnLSTM model which only modifies the test vectors additively,
    this model allows for an additive update to be performed to both test and
    support using information from each other.

    Parameters
    ----------
    n_support: int
      Size of support set.
    n_test: int
      Size of test set.
    n_feat: int
      Number of input atom features
    max_depth: int
      Number of LSTM Embedding layers.
    init: string
      Type of weight initialization (from Keras)
    activation: string
      Activation type (ReLu/Linear/etc.)
    """
    warnings.warn("The dc.nn.ResiLSTMEmbedding is "
                  "deprecated. Will be removed in DeepChem 1.4. "
                  "Will be replaced by "
                  "dc.models.tensorgraph.layers.IterRefLSTM",
                  DeprecationWarning)
    super(ResiLSTMEmbedding, self).__init__(**kwargs)

    self.init = initializations.get(init)  # Set weight initialization
    self.activation = activations.get(activation)  # Get activations
    self.max_depth = max_depth
    self.n_test = n_test
    self.n_support = n_support
    self.n_feat = n_feat

  #def build(self, input_shape):
  def build(self):
    """Builds this layer.
    """
    #_, support_input_shape = input_shape  #Unpack
    #n_feat = support_input_shape[1]
    n_feat = self.n_feat

    # Support set lstm
    self.support_lstm = LSTMStep(n_feat, 2 * n_feat)
    self.q_init = model_ops.zeros([self.n_support, n_feat])
    self.support_states_init = self.support_lstm.get_initial_states(
        [self.n_support, n_feat])

    # Test lstm
    self.test_lstm = LSTMStep(n_feat, 2 * n_feat)
    self.p_init = model_ops.zeros([self.n_test, n_feat])
    self.test_states_init = self.test_lstm.get_initial_states(
        [self.n_test, n_feat])

    self.trainable_weights = []

  def get_output_shape_for(self, input_shape):
    """Returns the output shape. Same as input_shape.

    Parameters
    ----------
    input_shape: list
      Will be of form [(n_test, n_feat), (n_support, n_feat)]

    Returns
    -------
    list
      Of same shape as input [(n_test, n_feat), (n_support, n_feat)]
    """
    return input_shape

  def call(self, argument, mask=None):
    """Execute this layer on input tensors.

    Parameters
    ----------
    argument: list
      List of two tensors (X, Xp). X should be of shape (n_test, n_feat) and
      Xp should be of shape (n_support, n_feat) where n_test is the size of
      the test set, n_support that of the support set, and n_feat is the number
      of per-atom features.

    Returns
    -------
    list
      Returns two tensors of same shape as input. Namely the output shape will
      be [(n_test, n_feat), (n_support, n_feat)]
    """
    self.build()
    x, xp = argument

    # Get initializations
    p = self.p_init
    q = self.q_init
    # Rename support
    z = xp
    states = self.support_states_init
    x_states = self.test_states_init

    for d in range(self.max_depth):
      # Process support xp using attention
      e = cos(z + q, xp)
      a = tf.nn.softmax(e)
      # Get linear combination of support set
      r = model_ops.dot(a, xp)

      # Not sure if it helps to place the update here or later yet.  Will
      # decide
      #z = r

      # Process test x using attention
      x_e = cos(x + p, z)
      x_a = tf.nn.softmax(x_e)
      s = model_ops.dot(x_a, z)

      # Generate new support attention states
      qr = model_ops.concatenate([q, r], axis=1)
      q, states = self.support_lstm([qr] + states)

      # Generate new test attention states
      ps = model_ops.concatenate([p, s], axis=1)
      p, x_states = self.test_lstm([ps] + x_states)

      # Redefine
      z = r

    #return [x+p, z+q]
    return [x + p, xp + q]

  def compute_mask(self, x, mask=None):
    if not (mask is None):
      return mask
    return [None, None]


def cos(x, y):
  denom = (
      model_ops.sqrt(model_ops.sum(tf.square(x)) * model_ops.sum(tf.square(y)))
      + model_ops.epsilon())
  return model_ops.dot(x, tf.transpose(y)) / denom


class LSTMStep(Layer):
  """ LSTM whose call is a single step in the LSTM.

  This layer exists because the Keras LSTM layer is intrinsically linked to an
  RNN with sequence inputs, and here, we will not be using sequence inputs, but
  rather we generate a sequence of inputs using the intermediate outputs of the
  LSTM, and so will require step by step operation of the lstm
  """

  def __init__(self,
               output_dim,
               input_dim,
               init='glorot_uniform',
               inner_init='orthogonal',
               forget_bias_init='one',
               activation='tanh',
               inner_activation='hard_sigmoid',
               **kwargs):

    warnings.warn("The dc.nn.LSTMStep is "
                  "deprecated. Will be removed in DeepChem 1.4. "
                  "Will be replaced by dc.models.tensorgraph.layers.LSTMStep",
                  DeprecationWarning)

    super(LSTMStep, self).__init__(**kwargs)

    self.output_dim = output_dim

    self.init = initializations.get(init)
    self.inner_init = initializations.get(inner_init)
    # No other forget biases supported right now.
    assert forget_bias_init == "one"
    self.forget_bias_init = initializations.get(forget_bias_init)
    self.activation = activations.get(activation)
    self.inner_activation = activations.get(inner_activation)
    self.input_dim = input_dim

  def get_initial_states(self, input_shape):
    return [model_ops.zeros(input_shape), model_ops.zeros(input_shape)]

  #def build(self, input_shape):
  def build(self):

    self.W = self.init((self.input_dim, 4 * self.output_dim))
    self.U = self.inner_init((self.output_dim, 4 * self.output_dim))

    self.b = tf.Variable(
        np.hstack((np.zeros(self.output_dim), np.ones(self.output_dim),
                   np.zeros(self.output_dim), np.zeros(self.output_dim))),
        dtype=tf.float32)
    self.trainable_weights = [self.W, self.U, self.b]

  def get_output_shape_for(self, input_shape):
    x, h_tm1, c_tm1 = input_shape  # Unpack
    return [(x[0], self.output_dim), h_tm1, c_tm1]

  def call(self, x_states, mask=None):
    self.build()
    x, h_tm1, c_tm1 = x_states  # Unpack

    # Taken from Keras code [citation needed]
    z = model_ops.dot(x, self.W) + model_ops.dot(h_tm1, self.U) + self.b

    z0 = z[:, :self.output_dim]
    z1 = z[:, self.output_dim:2 * self.output_dim]
    z2 = z[:, 2 * self.output_dim:3 * self.output_dim]
    z3 = z[:, 3 * self.output_dim:]

    i = self.inner_activation(z0)
    f = self.inner_activation(z1)
    c = f * c_tm1 + i * self.activation(z2)
    o = self.inner_activation(z3)

    h = o * self.activation(c)

    ####################################################### DEBUG
    #return o, [h, c]
    return h, [h, c]
    ####################################################### DEBUG


class DTNNEmbedding(Layer):
  """Generate embeddings for all atoms in the batch
  """

  def __init__(self,
               n_embedding=30,
               periodic_table_length=30,
               init='glorot_uniform',
               **kwargs):
    """
    Parameters
    ----------
    n_embedding: int, optional
      Number of features for each atom
    periodic_table_length: int, optional
      Length of embedding, 83=Bi
    init: str, optional
      Weight initialization for filters.
    """

    warnings.warn("The dc.nn.DTNNEmbedding is "
                  "deprecated. Will be removed in DeepChem 1.4. "
                  "Will be replaced by "
                  "dc.models.tensorgraph.graph_layers.DTNNEmbedding",
                  DeprecationWarning)
    self.n_embedding = n_embedding
    self.periodic_table_length = periodic_table_length
    self.init = initializations.get(init)  # Set weight initialization

    super(DTNNEmbedding, self).__init__(**kwargs)

  def build(self):

    self.embedding_list = self.init(
        [self.periodic_table_length, self.n_embedding])
    self.trainable_weights = [self.embedding_list]

  def call(self, x):
    """Execute this layer on input tensors.

    Parameters
    ----------
    x: Tensor
      1D tensor of length n_atoms (atomic number)

    Returns
    -------
    tf.Tensor
      Of shape (n_atoms, n_embedding), where n_embedding is number of atom features
    """
    self.build()
    atom_features = tf.nn.embedding_lookup(self.embedding_list, x)
    return atom_features


class DTNNStep(Layer):
  """A convolution step that merge in distance and atom info of
     all other atoms into current atom.

     model based on https://arxiv.org/abs/1609.08259
  """

  def __init__(self,
               n_embedding=30,
               n_distance=100,
               n_hidden=60,
               init='glorot_uniform',
               activation='tanh',
               **kwargs):
    """
    Parameters
    ----------
    n_embedding: int, optional
      Number of features for each atom
    n_distance: int, optional
      granularity of distance matrix
    n_hidden: int, optional
      Number of nodes in hidden layer
    init: str, optional
      Weight initialization for filters.
    activation: str, optional
      Activation function applied
    """
    warnings.warn("The dc.nn.DTNNStep is "
                  "deprecated. Will be removed in DeepChem 1.4. "
                  "Will be replaced by "
                  "dc.models.tensorgraph.graph_layers.DTNNStep",
                  DeprecationWarning)
    self.n_embedding = n_embedding
    self.n_distance = n_distance
    self.n_hidden = n_hidden
    self.init = initializations.get(init)  # Set weight initialization
    self.activation = activations.get(activation)  # Get activations

    super(DTNNStep, self).__init__(**kwargs)

  def build(self):
    self.W_cf = self.init([self.n_embedding, self.n_hidden])
    self.W_df = self.init([self.n_distance, self.n_hidden])
    self.W_fc = self.init([self.n_hidden, self.n_embedding])
    self.b_cf = model_ops.zeros(shape=[
        self.n_hidden,
    ])
    self.b_df = model_ops.zeros(shape=[
        self.n_hidden,
    ])
    #self.b_fc = model_ops.zeros(shape=[self.n_embedding,])

    self.trainable_weights = [
        self.W_cf, self.W_df, self.W_fc, self.b_cf, self.b_df
    ]

  def call(self, x):
    """Execute this layer on input tensors.

    Parameters
    ----------
    x: list of Tensor
      should be [atom_features: n_atoms*n_embedding,
                 distance_matrix: n_pairs*n_distance,
                 atom_membership: n_atoms
                 distance_membership_i: n_pairs,
                 distance_membership_j: n_pairs,
                 ]

    Returns
    -------
    tf.Tensor
      new embeddings for atoms, same shape as x[0]
    """
    self.build()
    atom_features = x[0]
    distance = x[1]
    distance_membership_i = x[3]
    distance_membership_j = x[4]
    distance_hidden = tf.matmul(distance, self.W_df) + self.b_df
    #distance_hidden = self.activation(distance_hidden)
    atom_features_hidden = tf.matmul(atom_features, self.W_cf) + self.b_cf
    #atom_features_hidden = self.activation(atom_features_hidden)
    outputs = tf.multiply(distance_hidden,
                          tf.gather(atom_features_hidden,
                                    distance_membership_j))

    # for atom i in a molecule m, this step multiplies together distance info of atom pair(i,j)
    # and embeddings of atom j(both gone through a hidden layer)
    outputs = tf.matmul(outputs, self.W_fc)
    outputs = self.activation(outputs)

    output_ii = tf.multiply(self.b_df, atom_features_hidden)
    output_ii = tf.matmul(output_ii, self.W_fc)
    output_ii = self.activation(output_ii)

    # for atom i, sum the influence from all other atom j in the molecule
    outputs = tf.segment_sum(outputs,
                             distance_membership_i) - output_ii + atom_features

    return outputs


class DTNNGather(Layer):
  """Map the atomic features into molecular properties and sum
  """

  def __init__(self,
               n_embedding=30,
               n_outputs=100,
               layer_sizes=[100],
               output_activation=True,
               init='glorot_uniform',
               activation='tanh',
               **kwargs):
    """
    Parameters
    ----------
    n_embedding: int, optional
      Number of features for each atom
    layer_sizes: list of int, optional(default=[1000])
      Structure of hidden layer(s)
    n_tasks: int, optional
      Number of final summed outputs
    init: str, optional
      Weight initialization for filters.
    activation: str, optional
      Activation function applied
    """
    warnings.warn("The dc.nn.DTNNGather is "
                  "deprecated. Will be removed in DeepChem 1.4. "
                  "Will be replaced by "
                  "dc.models.tensorgraph.graph_layers.DTNNGather",
                  DeprecationWarning)
    self.n_embedding = n_embedding
    self.layer_sizes = layer_sizes
    self.n_outputs = n_outputs
    self.output_activation = output_activation
    self.init = initializations.get(init)  # Set weight initialization
    self.activation = activations.get(activation)  # Get activations

    super(DTNNGather, self).__init__(**kwargs)

  def build(self):
    self.W_list = []
    self.b_list = []
    prev_layer_size = self.n_embedding
    for i, layer_size in enumerate(self.layer_sizes):
      self.W_list.append(self.init([prev_layer_size, layer_size]))
      self.b_list.append(model_ops.zeros(shape=[
          layer_size,
      ]))
      prev_layer_size = layer_size
    self.W_list.append(self.init([prev_layer_size, self.n_outputs]))
    self.b_list.append(model_ops.zeros(shape=[
        self.n_outputs,
    ]))

    self.trainable_weights = self.W_list + self.b_list

  def call(self, x):
    """Execute this layer on input tensors.

    Parameters
    ----------
    x: list of Tensor
      should be [embedding tensor of molecules, of shape (batch_size*max_n_atoms*n_embedding),
                 mask tensor of molecules, of shape (batch_size*max_n_atoms)]

    Returns
    -------
    list of tf.Tensor
      Of shape (batch_size)
    """
    self.build()
    output = x[0]
    atom_membership = x[1]
    for i, W in enumerate(self.W_list[:-1]):
      output = tf.matmul(output, W) + self.b_list[i]
      output = self.activation(output)
    output = tf.matmul(output, self.W_list[-1]) + self.b_list[-1]
    if self.output_activation:
      output = self.activation(output)
    output = tf.segment_sum(output, atom_membership)
    return output


class DAGLayer(Layer):
  """" Main layer of DAG model
  For a molecule with n atoms, n different graphs are generated and run through
  The final outputs of each graph become the graph features of corresponding
  atom, which will be summed and put into another network in DAGGather Layer
  """

  def __init__(self,
               n_graph_feat=30,
               n_atom_feat=75,
               max_atoms=50,
               layer_sizes=[100],
               init='glorot_uniform',
               activation='relu',
               dropout=None,
               batch_size=64,
               **kwargs):
    """
    Parameters
    ----------
    n_graph_feat: int, optional
      Number of features for each node(and the whole grah).
    n_atom_feat: int, optional
      Number of features listed per atom.
    max_atoms: int, optional
      Maximum number of atoms in molecules.
    layer_sizes: list of int, optional(default=[1000])
      Structure of hidden layer(s)
    init: str, optional
      Weight initialization for filters.
    activation: str, optional
      Activation function applied
    dropout: float, optional
      Dropout probability, not supported here
    batch_size: int, optional
      number of molecules in a batch
    """
    warnings.warn("The dc.nn.DAGLayer is "
                  "deprecated. Will be removed in DeepChem 1.4. "
                  "Will be replaced by "
                  "dc.models.tensorgraph.graph_layers.DAGLayer",
                  DeprecationWarning)
    super(DAGLayer, self).__init__(**kwargs)

    self.init = initializations.get(init)  # Set weight initialization
    self.activation = activations.get(activation)  # Get activations
    self.layer_sizes = layer_sizes
    self.dropout = dropout
    self.max_atoms = max_atoms
    self.batch_size = batch_size
    self.n_inputs = n_atom_feat + (self.max_atoms - 1) * n_graph_feat
    # number of inputs each step
    self.n_graph_feat = n_graph_feat
    self.n_outputs = n_graph_feat
    self.n_atom_feat = n_atom_feat

  def build(self):
    """"Construct internal trainable weights.
    """

    self.W_list = []
    self.b_list = []
    prev_layer_size = self.n_inputs
    for layer_size in self.layer_sizes:
      self.W_list.append(self.init([prev_layer_size, layer_size]))
      self.b_list.append(model_ops.zeros(shape=[
          layer_size,
      ]))
      prev_layer_size = layer_size
    self.W_list.append(self.init([prev_layer_size, self.n_outputs]))
    self.b_list.append(model_ops.zeros(shape=[
        self.n_outputs,
    ]))

    self.trainable_weights = self.W_list + self.b_list

  def call(self, x, mask=None):
    """Execute this layer on input tensors.

    x = [atom_features, parents, calculation_orders, calculation_masks, membership, n_atoms]

    Parameters
    ----------
    x: list
      list of Tensors of form described above.
    mask: bool, optional
      Ignored. Present only to shadow superclass call() method.

    Returns
    -------
    outputs: tf.Tensor
      Tensor of atom features, of shape (n_atoms, n_graph_feat)
    """
    # Add trainable weights
    self.build()
    # Extract atom_features
    # Basic features of every atom: (batch_size*max_atoms) * n_atom_features
    atom_features = x[0]
    # calculation orders of graph: (batch_size*max_atoms) * max_atoms * max_atoms
    # each atom corresponds to a graph, which is represented by the `max_atoms*max_atoms` int32 matrix of index
    # each gragh include `max_atoms` of steps(corresponding to rows) of calculating graph features
    # step i calculates the graph features for atoms of index `parents[:,i,0]`
    parents = x[1]
    # target atoms for each step: (batch_size*max_atoms) * max_atoms
    # represent the same atoms of `parents[:, :, 0]`,
    # different in that these index are positions in `atom_features`
    calculation_orders = x[2]
    calculation_masks = x[3]
    # number of atoms in total, should equal `batch_size*max_atoms`
    n_atoms = x[5]

    graph_features_initial = tf.zeros((self.max_atoms * self.batch_size,
                                       self.max_atoms + 1, self.n_graph_feat))
    # initialize graph features for each graph
    # another row of zeros is generated for padded dummy atoms
    graph_features = tf.Variable(graph_features_initial, trainable=False)

    for count in range(self.max_atoms):
      # `count`-th step
      # extracting atom features of target atoms: (batch_size*max_atoms) * n_atom_features
      mask = calculation_masks[:, count]
      current_round = tf.boolean_mask(calculation_orders[:, count], mask)
      batch_atom_features = tf.gather(atom_features, current_round)

      # generating index for graph features used in the inputs
      index = tf.stack(
          [
              tf.reshape(
                  tf.stack(
                      [tf.boolean_mask(tf.range(n_atoms), mask)] *
                      (self.max_atoms - 1),
                      axis=1), [-1]),
              tf.reshape(tf.boolean_mask(parents[:, count, 1:], mask), [-1])
          ],
          axis=1)
      # extracting graph features for parents of the target atoms, then flatten
      # shape: (batch_size*max_atoms) * [(max_atoms-1)*n_graph_features]
      batch_graph_features = tf.reshape(
          tf.gather_nd(graph_features, index),
          [-1, (self.max_atoms - 1) * self.n_graph_feat])

      # concat into the input tensor: (batch_size*max_atoms) * n_inputs
      batch_inputs = tf.concat(
          axis=1, values=[batch_atom_features, batch_graph_features])
      # DAGgraph_step maps from batch_inputs to a batch of graph_features
      # of shape: (batch_size*max_atoms) * n_graph_features
      # representing the graph features of target atoms in each graph
      batch_outputs = self.DAGgraph_step(batch_inputs, self.W_list, self.b_list)

      # index for targe atoms
      target_index = tf.stack([tf.range(n_atoms), parents[:, count, 0]], axis=1)
      target_index = tf.boolean_mask(target_index, mask)
      # update the graph features for target atoms
      graph_features = tf.scatter_nd_update(graph_features, target_index,
                                            batch_outputs)

    # last step generates graph features for all target atom
    return batch_outputs

  def DAGgraph_step(self, batch_inputs, W_list, b_list):
    outputs = batch_inputs
    for idw, W in enumerate(W_list):
      outputs = tf.nn.xw_plus_b(outputs, W, b_list[idw])
      outputs = self.activation(outputs)
    return outputs


class DAGGather(Layer):
  """ Gather layer of DAG model
  for each molecule, graph outputs are summed and input into another NN
  """

  def __init__(self,
               n_graph_feat=30,
               n_outputs=30,
               max_atoms=50,
               layer_sizes=[100],
               init='glorot_uniform',
               activation='relu',
               dropout=None,
               **kwargs):
    """
    Parameters
    ----------
    n_graph_feat: int, optional
      Number of features for each atom
    n_outputs: int, optional
      Number of features for each molecule.
    max_atoms: int, optional
      Maximum number of atoms in molecules.
    layer_sizes: list of int, optional
      Structure of hidden layer(s)
    init: str, optional
      Weight initialization for filters.
    activation: str, optional
      Activation function applied
    dropout: float, optional
      Dropout probability, not supported
    """
    warnings.warn("The dc.nn.DAGGather is "
                  "deprecated. Will be removed in DeepChem 1.4. "
                  "Will be replaced by "
                  "dc.models.tensorgraph.graph_layers.DAGGather",
                  DeprecationWarning)
    super(DAGGather, self).__init__(**kwargs)

    self.init = initializations.get(init)  # Set weight initialization
    self.activation = activations.get(activation)  # Get activations
    self.layer_sizes = layer_sizes
    self.dropout = dropout
    self.max_atoms = max_atoms
    self.n_graph_feat = n_graph_feat
    self.n_outputs = n_outputs

  def build(self):
    """"Construct internal trainable weights.
    """

    self.W_list = []
    self.b_list = []
    prev_layer_size = self.n_graph_feat
    for layer_size in self.layer_sizes:
      self.W_list.append(self.init([prev_layer_size, layer_size]))
      self.b_list.append(model_ops.zeros(shape=[
          layer_size,
      ]))
      prev_layer_size = layer_size
    self.W_list.append(self.init([prev_layer_size, self.n_outputs]))
    self.b_list.append(model_ops.zeros(shape=[
        self.n_outputs,
    ]))

    self.trainable_weights = self.W_list + self.b_list

  def call(self, x, mask=None):
    """Execute this layer on input tensors.

    x = [graph_features, membership]

    Parameters
    ----------
    x: tf.Tensor
      Tensor of each atom's graph features

    Returns
    -------
    outputs: tf.Tensor
      Tensor of each molecule's features

    """
    # Add trainable weights
    self.build()
    atom_features = x[0]
    membership = x[1]
    # Extract atom_features
    graph_features = tf.segment_sum(atom_features, membership)
    # sum all graph outputs
    outputs = self.DAGgraph_step(graph_features, self.W_list, self.b_list)
    return outputs

  def DAGgraph_step(self, batch_inputs, W_list, b_list):
    outputs = batch_inputs
    for idw, W in enumerate(W_list):
      outputs = tf.nn.xw_plus_b(outputs, W, b_list[idw])
      outputs = self.activation(outputs)
    return outputs
