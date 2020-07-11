# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 20:13:05 2020

@author: nitin
"""

# Build a model
inputs = Input(shape=(128,))
layer1 = Dense(64, activation='relu')(inputs)
layer2 = Dense(64, activation='relu')(layer1)
predictions = Dense(10, activation='softmax')(layer2)
model = Model(inputs=inputs, outputs=predictions)