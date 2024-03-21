# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from .detr_vae import build_act as build_act
from .detr_vae import build_interact as build_interact
from .detr_vae import build_cnnmlp as build_cnnmlp

def build_ACT_model(args):
    return build_act(args)

def build_InterACT_model(args):
    return build_interact(args)

def build_CNNMLP_model(args):
    return build_cnnmlp(args)