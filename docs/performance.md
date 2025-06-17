# Performance Analysis Report

## Overview

This document outlines latency and performance metrics observed for the Claude + ResNet Multi-Modal Analyzer. The service includes a local image classifier (ResNet 50) and a text analyzer using AWS Bedrock (Claude 3).

---

## Components Analyzed

| Component        | Description                              |
|------------------|------------------------------------------|
| Image Classifier | ResNet-50 running locally with PyTorch   |
| Text Processor   | Claude 3 via AWS Bedrock API             |
| UI               | Streamlit Interface (interactive chat)   |

---

## Environment

- CPU: MacBook
- RAM: 16 GB
- GPU: Not used
- OS: macOS

---

## Latency Metrics

| Action               | Average Time (s) | Notes                             |
|----------------------|------------------|-----------------------------------|
| Image Inference      | 1.75             | ResNet-50, Torch on CPU           |
| Claude 3 Completion  | 2.10             | Claude Sonnet, via AWS Bedrock    |
| Full Chat Turnaround | ~2.5             | Combined input-output cycle       |




