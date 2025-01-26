#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 14 12:30:55 2024

This fine is for initialization of the models' weight
@author: reza
"""

import math
import torch
import torch.nn as nn
from models.args import args


def signed_constant(module):
    fan = nn.init._calculate_correct_fan(module, args.mode)
    gain = nn.init.calculate_gain(args.nonlinearity)
    std = gain / math.sqrt(fan)
    module.data = module.data.sign() * std
    return module.data

def unsigned_constant(module):
    fan = nn.init._calculate_correct_fan(module, args.mode)
    gain = nn.init.calculate_gain(args.nonlinearity)
    std = gain / math.sqrt(fan)
    module.data = torch.ones_like(module.data) * std
    return module.data

def kaiming_normal(module):
    module.data = nn.init.kaiming_normal_(
        module, mode=args.mode, nonlinearity=args.nonlinearity
      )
    return module.data

def kaiming_uniform(module):
    module.data = nn.init.kaiming_uniform_(
        module, mode=args.mode, nonlinearity=args.nonlinearity
    )
    return module.data

def xavier_normal(module):
    module.data = kaiming_uniform = nn.init.xavier_normal_(
        module
    )
    return module.data

def glorot_uniform(module):
    module.data = nn.init.xavier_uniform_(
        module
    )
    return module.data

def xavier_constant(module):
    fan = nn.init._calculate_correct_fan(module, args.mode)
    gain = 1.0
    std = gain / math.sqrt(fan)
    module.data = module.data.sign() * std
    return module.data

def default(module):
    module.data = nn.init.kaiming_uniform_(module, a=math.sqrt(5))
    return module.data