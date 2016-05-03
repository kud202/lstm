--
----  Copyright (c) 2014, Facebook, Inc.
----  All rights reserved.
----
----  This source code is licensed under the Apache 2 license found in the
----  LICENSE file in the root directory of this source tree.
----

gpu = false
if gpu then
    require 'cunn'
    print("Running on GPU")

else
    require 'nn'
    print("Running on CPU")
end

require('nngraph')
require('base')
stringx = require('pl.stringx')
require 'io'
ptb = require('data')

-- Trains 1 epoch and gives validation set ~182 perplexity (CPU).
local params = {
                batch_size=10, -- minibatch
                seq_length=15, -- unroll length
                layers=2,
                decay=2,
                rnn_size=200, -- hidden unit size
                dropout=0.2,
                init_weight=0.1, -- random weight initialization limits
                lr=1, --learning rate
                vocab_size=10000, -- limit on the vocabulary size
                max_epoch=4,  -- when to start decaying learning rate
                max_max_epoch=13, -- final epoch
                max_grad_norm=5 -- clip when gradients exceed this norm value
               }

function transfer_data(x)
    if gpu then
        return x:cuda()
    else
        return x
    end
end

model = {}

--local function lstm(x, prev_c, prev_h)
  local function lstm(x, prev_h)
    --[[-- Calculate all four gates in one go
    --local i2h              = nn.Linear(params.rnn_size, 4*params.rnn_size)(x)
    local i2h              = nn.Linear(params.rnn_size, 2*params.rnn_size)(x)
    --local h2h              = nn.Linear(params.rnn_size, 4*params.rnn_size)(prev_h)
    local h2h              = nn.Linear(params.rnn_size, 2*params.rnn_size)(prev_h)
    local gates            = nn.CAddTable()({i2h, h2h})
    -- Reshape to (batch_size, n_gates, hid_size)
    -- Then slize the n_gates dimension, i.e dimension 2
    --local reshaped_gates   =  nn.Reshape(4,params.rnn_size)(gates)
    local reshaped_gates   =  nn.Reshape(2,params.rnn_size)(gates)
    local sliced_gates     = nn.SplitTable(2)(reshaped_gates)
    -- Use select gate to fetch each gate and apply nonlinearity
    local in_gate          = nn.Sigmoid()(nn.SelectTable(1)(sliced_gates))
    --local in_transform     = nn.Tanh()(nn.SelectTable(2)(sliced_gates))
    local forget_gate      = nn.Sigmoid()(nn.SelectTable(2)(sliced_gates))
    --local out_gate         = nn.Sigmoid()(nn.SelectTable(4)(sliced_gates))

    local new_h = nn.CMulTable()({forget_gate,prev_h})

    local i2newh    =nn.Linear(params.rnn_size , params.rnn_size)(x)
    local newh2newh =nn.Linear(params.rnn_size , params.rnn_size)(new_h)

    local transform_gate    =nn.CAddTable()({i2newh , newh2newh})

    local reshape_gate      =nn.Reshape(1,params.rnn_size)(transform_gate)
    local slice_gate        =nn.SplitTable(1)(reshape_gate)
    local in_transform      =nn.Tanh()(nn.SelectTable(1)(slice_gate))

    --local next_c           = nn.CAddTable()({
    --    nn.CMulTable()({forget_gate, prev_c}),
    --    nn.CMulTable()({in_gate,     in_transform})
    --})
    --local next_h           = nn.CMulTable()({out_gate, nn.Tanh()(next_c)})

    --local res=torch.Tensor():resizeAs(in_gate):fill(1)
    local res= nn.Copy(in_gate):fill(1)
    local newin_gate    = nn.CSubTable()({res,in_gate*-1})
    local next_h        = nn.CAddTable()({
        nn.CMulTable()({prev_h,newin_gate}),
        nn.CMulTable()({in_transform, newin_gate})
        })--]]

    local i2h   =nn.Linear(params.rnn_size, 3 *params.rnn_size)(x)

    local h2h   =nn.Linear(params.rnn_size, 3 * params.rnn_size)(prev_h)

    local gates =nn.CAddTable()({nn.Narrow(2, 1, 2 *params.rnn_size)(i2h),
                                nn.Narrow(2, 1, 2 *params.rnn_size)(h2h)
                                })

    gates =nn.SplitTable(2)(nn.Reshape(2,params.rnn_size)(gates))

    local resetgate  =nn.Sigmoid()(nn.SelectTable(1)(gates))

    local updategate =nn.Sigmoid()(nn.SelectTable(2)(gates))

    local output =nn.Tanh()(nn.CAddTable()({ nn.Narrow(2, 2 * params.rnn_size+1, params.rnn_size)(i2h),
                                            nn.CMulTable()({resetgate, nn.Narrow(2, 2 * params.rnn_size+1, params.rnn_size)(h2h)}) }))

    local next_h = nn.CAddTable()({ prev_h,nn.CMulTable()({ updategate, nn.CSubTable()({output, prev_h,}),}), })

    return next_h
end

function create_network()
    local x                  = nn.Identity()()
    local y                  = nn.Identity()()
    local prev_s             = nn.Identity()()
    local i                  = {[0] = nn.LookupTable(params.vocab_size,
                                                    params.rnn_size)(x)}
    local next_s             = {}
    --local split              = {prev_s:split(2 * params.layers)}
    local split              = {prev_s:split(params.layers)}
    for layer_idx = 1, params.layers do
        --local prev_c         = split[2 * layer_idx - 1]
        --local prev_h         = split[2 * layer_idx]
        local prev_h         = split[layer_idx]
        local dropped        = nn.Dropout(params.dropout)(i[layer_idx - 1])
        --local next_c, next_h = lstm(dropped, prev_c, prev_h)
        local next_h = lstm(dropped, prev_h)
        --table.insert(next_s, next_c)
        table.insert(next_s, next_h)
        i[layer_idx] = next_h
    end

    local h2y                = nn.Linear(params.rnn_size, params.vocab_size)
    local dropped            = nn.Dropout(params.dropout)(i[params.layers])
    local pred               = nn.LogSoftMax()(h2y(dropped))
    local err                = nn.ClassNLLCriterion()({pred, y})
    local module             = nn.gModule({x, y, prev_s},
                                      {err, nn.Identity()(next_s),pred})
    -- initialize weights
    module:getParameters():uniform(-params.init_weight, params.init_weight)
    return transfer_data(module)
end

function setup()
    print("Creating a RNN LSTM network.")
    local core_network = create_network()
    paramx, paramdx = core_network:getParameters()
    model.s = {}
    --model.pred={}
    model.ds = {}
    model.start_s = {}
    for j = 0, params.seq_length do
        model.s[j] = {}
        --for d = 1, 2 * params.layers do
        for d = 1, params.layers do
            model.s[j][d] = transfer_data(torch.zeros(params.batch_size, params.rnn_size))
            --model.pred[j][d]=transfer_data(torch.zeros(params.batch_size, params.rnn_size))
        end
    end
    --for d = 1, 2 * params.layers do
    for d = 1, params.layers do
        model.start_s[d] = transfer_data(torch.zeros(params.batch_size, params.rnn_size))
        model.ds[d] = transfer_data(torch.zeros(params.batch_size, params.rnn_size))
    end
    model.core_network = core_network
    model.rnns = g_cloneManyTimes(core_network, params.seq_length)
    model.norm_dw = 0
    model.pred= transfer_data(torch.zeros(params.seq_length))
    model.err = transfer_data(torch.zeros(params.seq_length))
end

function reset_state(state)
    state.pos = 1
    if model ~= nil and model.start_s ~= nil then
        --for d = 1, 2 * params.layers do
        for d = 1, params.layers do
            model.start_s[d]:zero()
        end
    end
end

function reset_ds()
    for d = 1, #model.ds do
        model.ds[d]:zero()
    end
end

function fp(state)
    -- g_replace_table(from, to).
    g_replace_table(model.s[0], model.start_s)

    -- reset state when we are done with one full epoch
    if state.pos + params.seq_length > state.data:size(1) then
        reset_state(state)
    end

    -- forward prop
    for i = 1, params.seq_length do
        local x = state.data[state.pos]
        --print(x:size())
        local y = state.data[state.pos + 1]
        --print(y:size())
        local s = model.s[i - 1]
        --print(model.rnns[i]:forward({x, y, s}):size())
        model.err[i], model.s[i],pred = unpack(model.rnns[i]:forward({x, y, s}))
        state.pos = state.pos + 1
    end

    --print(model.pred)

    -- next-forward-prop start state is current-forward-prop's last state
    g_replace_table(model.start_s, model.s[params.seq_length])

    -- cross entropy error
    return model.err:mean()
end

function bp(state)
    -- start on a clean slate. Backprop over time for params.seq_length.
    paramdx:zero()
    reset_ds()
    --local pred = torch.zeros(params.seq_length)
    for i = params.seq_length, 1, -1 do
        -- to make the following code look almost like fp
        state.pos = state.pos - 1
        local x = state.data[state.pos]
        local y = state.data[state.pos + 1]
        local s = model.s[i - 1]
        -- Why 1?

        local derr = transfer_data(torch.ones(1))
        local dpred = transfer_data(torch.zeros(params.batch_size, params.vocab_size))
        -- tmp stores the ds
        local tmp = model.rnns[i]:backward({x, y, s},
                                           {derr, model.ds,dpred})[3]
        -- remember (to, from)
        g_replace_table(model.ds, tmp)
    end

    -- undo changes due to changing position in bp
    state.pos = state.pos + params.seq_length

    -- gradient clipping
    model.norm_dw = paramdx:norm()
    if model.norm_dw > params.max_grad_norm then
        local shrink_factor = params.max_grad_norm / model.norm_dw
        paramdx:mul(shrink_factor)
    end

    -- gradient descent step
    paramx:add(paramdx:mul(-params.lr))
end

function run_valid()
    -- again start with a clean slate
    reset_state(state_valid)

    -- no dropout in testing/validating
    g_disable_dropout(model.rnns)

    -- collect perplexity over the whole validation set
    local len = (state_valid.data:size(1) - 1) / (params.seq_length)
    local perp = 0
    for i = 1, len do
        perp = perp + fp(state_valid)
    end
    print("Validation set perplexity : " .. g_f3(torch.exp(perp / len)))
    g_enable_dropout(model.rnns)
end

function run_test()
    reset_state(state_test)
    g_disable_dropout(model.rnns)
    local perp = 0
    local len = state_test.data:size(1)

    -- no batching here
    g_replace_table(model.s[0], model.start_s)
    for i = 1, (len - 1) do
        local x = state_test.data[i]
        local y = state_test.data[i + 1]
        perp_tmp, model.s[1] = unpack(model.rnns[1]:forward({x, y, model.s[0]}))
        perp = perp + perp_tmp[1]
        g_replace_table(model.s[0], model.s[1])
    end
    print("Test set perplexity : " .. g_f3(torch.exp(perp / (len - 1))))
    g_enable_dropout(model.rnns)
end

function readline()
  local line = io.read("*line")
  if line == nil then error({code="EOF"}) end
  line = stringx.split(line)
  if tonumber(line[1]) == nil then error({code="init"}) end
  for i = 2,#line do
    if ptb.vocab_map[line[i]] == nil then error({code="vocab", word = line[i]}) end
  end
  return line
end

function run_query()
    local state_query = {data={}}
    while true do
      print("Query: len word1 word2 etc")
      local ok, line = pcall(readline)
      if not ok then
        if line.code == "EOF" then
          break -- end loop
        elseif line.code == "vocab" then
          print("Word not in vocabulary, only 'foo' is in vocabulary: ", line.word)
        elseif line.code == "init" then
          print("Start with a number")
        else
          print(line)
          print("Failed, try again")
        end
      else
        for i = 2,#line do
          state_query.data[i-1] = ptb.vocab_map[line[i]]
          io.write(line[i])
          io.write(' ')
        end
        reset_state(state_query)
        g_disable_dropout(model.rnns)
        g_replace_table(model.s[0], model.start_s)
        x = line[#line-1]
        y = line[#line]
        for i = 1, line[1] do
          perp_tmp, model.s[1],pred = unpack(model.rnns[1]:forward({x, y, model.s[0]}))
          g_replace_table(model.s[0], model.s[1])
          x = y
          y = int(pred)
          io.write(ptb.vocab_inv_map(y))
          io.write(' ')
        end
        io.write('\n')
      end
    end
end

if gpu then
    g_init_gpu(arg)
end

-- get data in batches
state_train = {data=transfer_data(ptb.traindataset(params.batch_size))}
state_valid =  {data=transfer_data(ptb.validdataset(params.batch_size))}
state_test =  {data=transfer_data(ptb.testdataset(params.batch_size))}

print("Network parameters:")
print(params)

local states = {state_train, state_valid, state_test}
for _, state in pairs(states) do
    reset_state(state)
end
setup()
run_query()
--[[
step = 0
epoch = 0
total_cases = 0
beginning_time = torch.tic()
start_time = torch.tic()
print("Starting training.")
words_per_step = params.seq_length * params.batch_size
epoch_size = torch.floor(state_train.data:size(1) / params.seq_length)

--print(state_train.data:size())

while epoch < params.max_max_epoch do

    -- take one step forward
    perp = fp(state_train)
    if perps == nil then
        perps = torch.zeros(epoch_size):add(perp)
    end
    perps[step % epoch_size + 1] = perp
    step = step + 1

    -- gradient over the step
    bp(state_train)

    -- words_per_step covered in one step
    total_cases = total_cases + params.seq_length * params.batch_size
    epoch = step / epoch_size

    -- display details at some interval
    if step % torch.round(epoch_size / 10) == 10 then
        wps = torch.floor(total_cases / torch.toc(start_time))
        since_beginning = g_d(torch.toc(beginning_time) / 60)
        print('epoch = ' .. g_f3(epoch) ..
             ', train perp. = ' .. g_f3(torch.exp(perps:mean())) ..
             ', wps = ' .. wps ..
             ', dw:norm() = ' .. g_f3(model.norm_dw) ..
             ', lr = ' ..  g_f3(params.lr) ..
             ', since beginning = ' .. since_beginning .. ' mins.')

        print('saving core')
        torch.save('core.net',model.core_network)
        print('saved core')
    end

    -- run when epoch done
    if step % epoch_size == 0 then
        run_valid()
        if epoch > params.max_epoch then
            params.lr = params.lr / params.decay
        end
    end
end
run_test()
print("Training is over.")
]]--
