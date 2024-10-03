# -*- coding: utf-8 -*-
"""
Created on Fri Sep 22 18:35:48 2023

@author: fsagir
"""
import robin_stocks.robinhood as rs
def main ():
    rs.login(username="fasil.sagir@outlook.com",
             password="Thebestf*123",
             expiresIn=86400,
             by_sms=True)