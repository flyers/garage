import tensorflow as tf

from garage.core import Serializable
from garage.tf.core import LayersPowered
import garage.tf.core.layers as layers
from garage.tf.core.layers import batch_norm
from garage.tf.misc import tensor_utils
from garage.tf.q_functions import QFunction


class ContinuousConvQFunction(QFunction, LayersPowered, Serializable):
    """
    This class implements a q value network to predict q based on the input
    state and action. It uses a network with convolutional layers to fit the function of Q(s, a).
    """

    def __init__(self,
                 env_spec,
                 conv_filters,
                 conv_filter_sizes,
                 conv_strides,
                 conv_pads,
                 hidden_sizes=(32, 32),
                 pool_size=2,
                 name="ContinuousConvQFunc",
                 hidden_nonlinearity=tf.nn.relu,
                 output_nonlinearity=None,
                 action_merge_layer=-2,
                 input_include_goal=False,
                 weight_normalization=False,
                 pooling=False,
                 bn=False):
        """
        Initialize class with multiple attributes.

        Args:
            env_spec():
            name(str, optional): A str contains the name of the policy.
            hidden_sizes(list or tuple, optional):
                A list of numbers of hidden units for all hidden layers.
            hidden_nonlinearity(optional):
                An activation shared by all fc layers.
            action_merge_layer(int, optional):
                An index to indicate when to merge action layer.
            output_nonlinearity(optional):
                An activation used by the output layer.
            bn(bool, optional):
                A bool to indicate whether normalize the layer or not.
        """
        Serializable.quick_init(self, locals())

        self.name = name
        self._env_spec = env_spec
        if input_include_goal:
            self._obs_dim = env_spec.observation_space.flat_dim_with_keys(
                ["observation", "desired_goal"])
        else:
            self._obs_dim = env_spec.observation_space.flat_dim
        self._action_dim = env_spec.action_space.flat_dim
        self._conv_filters = conv_filters
        self._conv_filter_sizes = conv_filter_sizes
        self._conv_strides = conv_strides
        self._conv_pads = conv_pads
        self._hidden_sizes = hidden_sizes
        self._hidden_nonlinearity = hidden_nonlinearity
        self._action_merge_layer = action_merge_layer
        self._output_nonlinearity = output_nonlinearity
        self._batch_norm = bn
        self._weight_normalization = weight_normalization
        self._pooling = pooling
        self._pool_size = pool_size
        self._f_qval, self._output_layer, self._obs_layer, self._action_layer = self.build_net(  # noqa: E501
            name=self.name)
        LayersPowered.__init__(self, [self._output_layer])

    def build_net(self, trainable=True, name=None):
        """
        Set up q network based on class attributes. This function uses layers
        defined in garage.tf.

        Args:
            reuse: A bool indicates whether reuse variables in the same scope.
            trainable: A bool indicates whether variables are trainable.
        """
        input_shape = self._env_spec.observation_space.shape
        assert len(input_shape) in [2, 3]
        if len(input_shape) == 2:
            input_shape = (1, ) + input_shape

        with tf.variable_scope(name):
            l_in = layers.InputLayer(shape=(None, self._obs_dim), name="obs")
            l_hid = layers.reshape(
                l_in, ([0], ) + input_shape, name="reshape_input")

            if self._batch_norm:
                l_hid = layers.batch_norm(l_hid)

            for idx, conv_filter, filter_size, stride, pad in zip(
                    range(len(self._conv_filters)),
                    self._conv_filters,
                    self._conv_filter_sizes,
                    self._conv_strides,
                    self._conv_pads,
            ):
                l_hid = layers.Conv2DLayer(
                    l_hid,
                    num_filters=conv_filter,
                    filter_size=filter_size,
                    stride=(stride, stride),
                    pad=pad,
                    nonlinearity=self._hidden_nonlinearity,
                    name="conv_hidden_%d" % idx,
                    weight_normalization=self._weight_normalization,
                    trainable=trainable,
                )
                if self._pooling:
                    l_hid = layers.Pool2DLayer(
                        l_hid, pool_size=self._pool_size)
                if self._batch_norm:
                    l_hid = layers.batch_norm(l_hid)

            l_hid = layers.flatten(l_hid, name="conv_flatten")
            l_action = layers.InputLayer(
                shape=(None, self._action_dim), name="actions")

            n_layers = len(self._hidden_sizes) + 1
            if n_layers > 1:
                action_merge_layer = \
                    (self._action_merge_layer % n_layers + n_layers) % n_layers
            else:
                action_merge_layer = 1

            for idx, size in enumerate(self._hidden_sizes):
                if self._batch_norm:
                    l_hid = batch_norm(l_hid)

                if idx == action_merge_layer:
                    l_hid = layers.ConcatLayer([l_hid, l_action])

                l_hid = layers.DenseLayer(
                    l_hid,
                    num_units=size,
                    nonlinearity=self._hidden_nonlinearity,
                    trainable=trainable,
                    name="hidden_%d" % (idx + 1))

            if action_merge_layer == n_layers:
                l_hid = layers.ConcatLayer([l_hid, l_action])

            l_output = layers.DenseLayer(
                l_hid,
                num_units=1,
                nonlinearity=self._output_nonlinearity,
                trainable=trainable,
                name="output")

            output_var = layers.get_output(l_output)

        f_qval = tensor_utils.compile_function(
            [l_in.input_var, l_action.input_var], output_var)
        output_layer = l_output
        obs_layer = l_in
        action_layer = l_action

        return f_qval, output_layer, obs_layer, action_layer

    def get_qval(self, observations, actions):
        return self._f_qval(observations, actions)

    def get_qval_sym(self, obs_var, action_var, name=None, **kwargs):
        with tf.name_scope(name, "get_qval_sym", values=[obs_var, action_var]):
            qvals = layers.get_output(self._output_layer, {
                self._obs_layer: obs_var,
                self._action_layer: action_var
            }, **kwargs)
            return qvals
