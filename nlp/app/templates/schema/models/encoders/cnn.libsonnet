// @file cnn.libsonnet
// @brief
// @author QRS
// @version 1.0
// @date 2020-05-22 22:49

local _Utils = import '../../utils/helper.libsonnet';

{
    get(jid):: {
        type: 'H',
        objs: [
            _Utils.int(
                jid + '.embedding_dim',
                'cnn dim',
                def=16,
                tips='this is the input dimension to the encoder'
            ),
            _Utils.int(
                jid + '.num_filters',
                'num filters',
                def=128,
                tips='this is the output dim for each convolutional layer, which is the number of filters learned by that layer.'
            ),
            _Utils.intarray(
                jid + '.ngram_filter_sizes',
                'ngram sizes',
                def=[2, 3, 4, 5],
                tips='This specifies both the number of convolutional layers we will create and their sizes'
            ),
            _Utils.string(
                jid + '.conv_layer_activation',
                'activation',
                def='relu',
                readonly=true,
                tips='activation to use after the convolution layers',
            ),
        ],
    },
}
