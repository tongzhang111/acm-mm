import math
import torch
import torch.nn as nn


def mask_logits(inputs, mask, mask_value=-1e30):
    mask = mask.type(torch.float32)
    return inputs + (mask) * mask_value

def transpose_for_score(input_tensor, batch_s, seq_length, num_heads, head_size):
    output_tensor = input_tensor.reshape(batch_s, seq_length, num_heads, head_size)
    output_tensor = output_tensor.transpose(1,2)
    return output_tensor


def create_attention_mask(from_mask, to_mask, broadcast_ones=False):
    bs, from_seq_len = from_mask.float().size()[0], from_mask.size()[1]
    # to_seq_len = to_mask.size()[1]
    to_mask = to_mask.float().unsqueeze(1)
    if broadcast_ones:
        mask = torch.ones((bs, from_seq_len, 1))
    else:
        mask = from_mask.unsqueeze(dim=2)
    mask = torch.mul(mask, to_mask)
    return mask

def layer_norm(inputs):
    dim = inputs.size()
    scale = torch.ones(dim[-1]).cuda()
    bias = torch.zeros(dim[-1]).cuda()
    mean = torch.mean(inputs, dim=-1, keepdim=True)
    variance = torch.mean(torch.square(inputs - mean), dim=-1, keepdim=True)
    norm_inputs = (inputs - mean) * torch.rsqrt(variance + 1e-6)
    results = norm_inputs * scale + bias
    return results

# def conv1d(inputs, dim, kernel_size=1, use_bias=False, activation=None, padding='VALID', reuse=None):
#     shapes = inputs.size()
#     # kernel = torch.tensor((kernel_size, shapes[-1], dim))
#     kernel = kernel_size
#     # outputs = torch.nn.Conv1d(inputs, kernel, stride=1, padding=padding)
#     conv = torch.nn.Conv1d(shapes[-1], shapes[-1], kernel, stride=1).cuda()
#     inputs = inputs.permute(0,2,1)
#     outputs = conv(inputs)
#     outputs = outputs.permute(0,2,1)
#     if use_bias:
#         bias = torch.zeros((1,1,dim)).cuda()
#         outputs += bias
#     return outputs if activation is None else activation(outputs)

def bilinear(input1, input2, dim, use_bias=True):
    conv1d_1 = Conv1D(in_dim=dim, out_dim=dim)
    conv1d_2 = Conv1D(in_dim=dim, out_dim=dim)

    input1 = conv1d_1(input1)
    input2 = conv1d_2(input2)
    output = input1 + input2
    if use_bias:
        bias = torch.zeros((1, 1, dim)).cuda()
        output += bias
    return output


class dual_multihead_attention(nn.Module):
    def __init__(self, dim, num_heads, drop_rate):
        super(dual_multihead_attention, self).__init__()
        assert dim % num_heads == 0, 'The channels (%d) is not a multiple of attention heads (%d)' % (dim, num_heads)
        self.head_size, self.num_heads, self.dim = int(dim / num_heads), num_heads, dim
        self.dropout = nn.Dropout(p=drop_rate)
        # def dual_multihead_attention(from_tensor, to_tensor, dim, num_heads, from_mask, to_mask, drop_rate=0, reuse=None):
        # drop = torch.nn.Dropout(drop_rate)
        self.query = Conv1D(in_dim=dim, out_dim=dim, kernel_size=1, stride=1, padding=0, bias=True)
        self.f_key = Conv1D(in_dim=dim, out_dim=dim, kernel_size=1, stride=1, padding=0, bias=True)
        self.f_value = Conv1D(in_dim=dim, out_dim=dim, kernel_size=1, stride=1, padding=0, bias=True)
        self.t_key = Conv1D(in_dim=dim, out_dim=dim, kernel_size=1, stride=1, padding=0, bias=True)
        self.t_value = Conv1D(in_dim=dim, out_dim=dim, kernel_size=1, stride=1, padding=0, bias=True)
        self.s_value = Conv1D(in_dim=dim, out_dim=dim, kernel_size=1, stride=1, padding=0, bias=True)
        self.x_value = Conv1D(in_dim=dim, out_dim=dim, kernel_size=1, stride=1, padding=0, bias=True)
        self.s_score= Conv1D(in_dim=dim, out_dim=dim, kernel_size=1, stride=1, padding=0, bias=True, activation=torch.sigmoid)
        self.x_score= Conv1D(in_dim=dim, out_dim=dim, kernel_size=1, stride=1, padding=0, bias=True, activation=torch.sigmoid)
        self.output = Conv1D(in_dim=dim, out_dim=dim, kernel_size=1, stride=1, padding=0, bias=True)
        # if dim % num_heads != 0:
        # raise ValueError('The hidden size (%d) is not a multiple of the attention heads (%d)' % (dim, num_heads))
        # bs, from_seq_len, to_seq_len = from_tensor.size()[0], from_tensor.size()[1], to_tensor.size()[1]
        # head_size = dim // num_heads


    def forward(self, from_tensor, to_tensor, from_mask, to_mask):
        bs, from_seq_len, to_seq_len = from_tensor.size()[0], from_tensor.size()[1], to_tensor.size()[1]
        query = transpose_for_score(self.query(from_tensor), bs, from_seq_len, self.num_heads, self.head_size)
        f_key = transpose_for_score(self.f_key(from_tensor), bs, from_seq_len, self.num_heads, self.head_size)
        f_value = transpose_for_score(self.f_value(from_tensor), bs, from_seq_len, self.num_heads, self.head_size)

        t_key = transpose_for_score(self.t_key(to_tensor), bs, to_seq_len, self.num_heads, self.head_size)
        t_value = transpose_for_score(self.t_value(to_tensor), bs, to_seq_len, self.num_heads, self.head_size)

        # create attention mask
        s_attn_mask = torch.unsqueeze(create_attention_mask(from_mask, from_mask, broadcast_ones=False), dim=1)
        x_attn_mask = torch.unsqueeze(create_attention_mask(from_mask, to_mask, broadcast_ones=False), dim=1)
        # compute self-attn score

        s_attn_value = torch.multiply(torch.matmul(query, f_key.transpose(2,3)), 1.0 / math.sqrt(float(self.head_size)))
        s_attn_value += (s_attn_mask) * -1e30
        s_attn_score = torch.softmax(s_attn_value, dim=-1)
        s_attn_score = self.dropout(s_attn_score)

        # compute cross-attn score
        x_attn_value = torch.multiply(torch.matmul(query, t_key.transpose(2,3)), 1.0 / math.sqrt(float(self.head_size)))
        x_attn_value += (x_attn_mask) * -1e30
        x_attn_score = torch.softmax(x_attn_value, dim=-1)
        x_attn_score =self.dropout(x_attn_score)

        # compute self-attn value
        s_value = torch.matmul(s_attn_score, f_value).permute([0, 2, 1, 3])
        s_value = s_value.reshape(bs, from_seq_len, self.num_heads * self.head_size)
        s_value = self.s_value(s_value)
        # compute cross-attn value
        x_value = torch.matmul(x_attn_score, t_value).permute([0, 2, 1, 3])
        x_value = x_value.reshape(bs, from_seq_len, self.num_heads * self.head_size)
        x_value = self.x_value(x_value)

        s_score = self.s_score(s_value)
        x_score = self.x_score(x_value)
        outputs = torch.multiply(s_score, x_value) + torch.multiply(x_score, s_value)

        outputs = self.output(outputs)
        scores = bilinear(from_tensor, outputs, dim=self.dim, use_bias=True)
        values = bilinear(from_tensor, outputs, dim=self.dim, use_bias=True)
        outputs = torch.sigmoid(mask_logits(scores, from_mask.unsqueeze(2))) * values
        return outputs

class dual_attn_bloack(nn.Module):
    def __init__(self, dim, num_heads, drop_rate):
        super(dual_attn_bloack, self).__init__()
        self.dual_multihead_attention = dual_multihead_attention(dim, num_heads, drop_rate)
        self.outputs = Conv1D(in_dim=dim, out_dim=dim)
        self.drop = torch.nn.Dropout(drop_rate)
        self.outputs_2 = Conv1D(in_dim=dim, out_dim=dim)

        # def dual_attn_bloack_seg(from_tensor, to_tensor, dim, num_heads, from_mask, to_mask, drop_rate, use_bias=True,
        #                      activation=None, reuse=None):
    def forward(self,from_tensor, to_tensor, from_mask, to_mask):

        outputs = layer_norm(from_tensor)
        to_tensor = layer_norm(to_tensor)
        outputs = self.dual_multihead_attention(from_tensor=outputs, to_tensor=to_tensor, from_mask=from_mask, to_mask=to_mask)
        outputs = self.outputs(outputs)
        residual = self.drop(outputs) + from_tensor

        outputs = layer_norm(residual)
        outputs = self.drop(outputs)
        outputs = self.outputs_2(outputs)
        outputs = self.drop(outputs) + residual
        return outputs


class t_attention(nn.Module):
    def __init__(self, dim, num_heads, drop_rate):
        super(t_attention, self).__init__()
        self.dual_multihead_attention = nn.MultiheadAttention(dim, num_heads, dropout=drop_rate)
        self.outputs = Conv1D(in_dim=dim, out_dim=dim)
        self.drop = torch.nn.Dropout(drop_rate)
        # def dual_attn_bloack_seg(from_tensor, to_tensor, dim, num_heads, from_mask, to_mask, drop_rate, use_bias=True,
        #                      activation=None, reuse=None):
    def forward(self,from_tensor, to_tensor, from_mask, to_mask):
        from_tensor = from_tensor.transpose(0,1)
        outputs = layer_norm(from_tensor)
        to_tensor = layer_norm(to_tensor)
        outputs = self.dual_multihead_attention(outputs,to_tensor,to_tensor)[0]
        outputs = self.outputs(outputs)
        residual = self.drop(outputs) + from_tensor

        outputs = layer_norm(residual)
        outputs = self.drop(outputs)
        outputs = self.outputs_2(outputs)
        outputs = self.drop(outputs) + residual
        return outputs


class t_cross_weight(nn.Module):
    def __init__(self, dim, num_heads, drop_rate):
        super(t_cross_weight, self).__init__()
        self.dual_multihead_attention = nn.MultiheadAttention(dim, num_heads, dropout=drop_rate)
        self.outputs = Conv1D(in_dim=dim, out_dim=dim)
        self.drop = torch.nn.Dropout(drop_rate)
        self.outputs_2 = Conv1D(in_dim=dim, out_dim=dim)
        w4C = torch.ones(100, 1)
        self.w4C = nn.Parameter(w4C, requires_grad=True)

        # def dual_attn_bloack_seg(from_tensor, to_tensor, dim, num_heads, from_mask, to_mask, drop_rate, use_bias=True,
        #                      activation=None, reuse=None):
    def forward(self,from_tensor, to_tensor, src):
        outputs = layer_norm(from_tensor)
        to_tensor = layer_norm(to_tensor)
        sim_score = torch.bmm(outputs,to_tensor.transpose(1,2))
        # sim_score = torch.softmax(sim_score,dim=1)   #是否考虑softmax变成sigmoid
        sim_score = torch.sigmoid(sim_score)   #是否考虑softmax变成sigmoid
        outputs = self.drop(sim_score)
        outputs = torch.multiply(outputs,src)
        #outputs = (torch.tanh(self.w4C[:from_tensor.shape[1]])+1).unsqueeze(0) * outputs
        residual = self.drop(outputs) + src

        outputs = layer_norm(residual)
        outputs = self.drop(outputs)
        outputs = self.outputs_2(outputs)
        outputs = self.drop(outputs) + src
        return outputs


class CQAttention(nn.Module):
    def __init__(self, dim, drop_rate=0.0):
        super(CQAttention, self).__init__()
        w4C = torch.empty(dim, 1)
        w4Q = torch.empty(dim, 1)
        w4mlu = torch.empty(1, 1, dim)
        nn.init.xavier_uniform_(w4C)
        nn.init.xavier_uniform_(w4Q)
        nn.init.xavier_uniform_(w4mlu)
        self.w4C = nn.Parameter(w4C, requires_grad=True)
        self.w4Q = nn.Parameter(w4Q, requires_grad=True)
        self.w4mlu = nn.Parameter(w4mlu, requires_grad=True)
        self.dropout = nn.Dropout(p=drop_rate)
        self.cqa_linear = Conv1D(in_dim=4 * dim, out_dim=dim, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, context, query, c_mask, q_mask):
        score = self.trilinear_attention(context, query)  # (batch_size, c_seq_len, q_seq_len)
        score_ = nn.Softmax(dim=2)(mask_logits(score, q_mask.unsqueeze(1)))  # (batch_size, c_seq_len, q_seq_len)
        score_t = nn.Softmax(dim=1)(mask_logits(score, c_mask.unsqueeze(2)))  # (batch_size, c_seq_len, q_seq_len)
        score_t = score_t.transpose(1, 2)  # (batch_size, q_seq_len, c_seq_len)
        c2q = torch.matmul(score_, query)  # (batch_size, c_seq_len, dim)
        q2c = torch.matmul(torch.matmul(score_, score_t), context)  # (batch_size, c_seq_len, dim)
        output = torch.cat([context, c2q, torch.mul(context, c2q), torch.mul(context, q2c)], dim=2)
        output = self.cqa_linear(output)  # (batch_size, c_seq_len, dim)
        return output

    def trilinear_attention(self, context, query):
        batch_size, c_seq_len, dim = context.shape
        batch_size, q_seq_len, dim = query.shape
        context = self.dropout(context)
        query = self.dropout(query)
        subres0 = torch.matmul(context, self.w4C).expand([-1, -1, q_seq_len])  # (batch_size, c_seq_len, q_seq_len)
        subres1 = torch.matmul(query, self.w4Q).transpose(1, 2).expand([-1, c_seq_len, -1])
        subres2 = torch.matmul(context * self.w4mlu, query.transpose(1, 2))
        res = subres0 + subres1 + subres2  # (batch_size, c_seq_len, q_seq_len)
        return res

class CQConcatenate(nn.Module):
    def __init__(self, dim):
        super(CQConcatenate, self).__init__()
        self.weighted_pool = WeightedPool(dim=dim)
        self.conv1d = Conv1D(in_dim=2 * dim, out_dim=dim, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, context, query, q_mask):
        pooled_query = self.weighted_pool(query, q_mask)  # (batch_size, dim)
        _, c_seq_len, _ = context.shape
        pooled_query = pooled_query.unsqueeze(1).repeat(1, c_seq_len, 1)  # (batch_size, c_seq_len, dim)
        output = torch.cat([context, pooled_query], dim=2)  # (batch_size, c_seq_len, 2*dim)
        output = self.conv1d(output)
        return output

class WeightedPool(nn.Module):
    def __init__(self, dim):
        super(WeightedPool, self).__init__()
        weight = torch.empty(dim, 1)
        nn.init.xavier_uniform_(weight)
        self.weight = nn.Parameter(weight, requires_grad=True)

    def forward(self, x, mask):
        alpha = torch.tensordot(x, self.weight, dims=1)  # shape = (batch_size, seq_length, 1)
        alpha = mask_logits(alpha, mask=mask.unsqueeze(2))
        alphas = nn.Softmax(dim=1)(alpha)
        pooled_x = torch.matmul(x.transpose(1, 2), alphas)  # (batch_size, dim, 1)
        pooled_x = pooled_x.squeeze(2)
        return pooled_x

class Conv1D(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size=1, stride=1, padding=0, bias=True, activation=None):
        super(Conv1D, self).__init__()
        self.conv1d = nn.Conv1d(in_channels=in_dim, out_channels=out_dim, kernel_size=kernel_size, padding=padding,
                                stride=stride, bias=bias).cuda()
        self.activation = activation
    def forward(self, x):
        # suppose all the input with shape (batch_size, seq_len, dim)
        x = x.transpose(1, 2)  # (batch_size, dim, seq_len)
        x = self.conv1d(x).transpose(1,2)
        return x if self.activation is None else self.activation(x)  # (batch_size, seq_len, dim)









