"""Capability evals: does the model still reason, still call tools, and talk like a pirate?

These exist to answer one question before and after every fine-tune: did adding
the pirate voice cost anything? A style SFT that quietly halves GSM8K is the
expected failure mode, not an exotic one, so the gates run on both sides of it.
"""
