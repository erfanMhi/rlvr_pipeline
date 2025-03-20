# Infrastructure Guide: GPU Resources & Extensibility

## Overview

This repository contains Terraform configurations for provisioning infrastructure resources on different cloud providers. Currently, it supports Nebius Cloud for both single GPU and clustered GPU resources. The structure is designed to be easily extensible for other cloud providers.

## Available Configurations

### Nebius Cloud

- **Single GPU Setup** - For individual GPU workloads
- **GPU Cluster Setup** - For distributed computing and training

## Setting Up GPU Resources

### Prerequisites

- [Terraform](https://www.terraform.io/downloads.html) (v1.0.0+)
- [Nebius CLI](https://nebius.ai/docs/cli-tools/) (for Nebius resources)
- Valid Nebius Cloud account credentials
- SSH key pair

### Nebius Single GPU Setup

1. Navigate to the single GPU configuration directory:
   ```bash
   cd infrastructure/nebius/single_gpu
   ```

2. Create your configuration file from the template:
   ```bash
   cp terraform.tfvars.example terraform.tfvars
   ```

3. Edit the `terraform.tfvars` file with your credentials and preferences:
   ```bash
   nano terraform.tfvars
   ```

   Configure the following parameters:
   - Nebius credentials (tenant_id, project_id)
   - GPU type and model
   - Network configuration
   - SSH key path

4. Initialize and apply the configuration:
   ```bash
   terraform init
   terraform plan
   terraform apply
   ```

5. Connect to your GPU instance using the output connection information:
   ```bash
   ssh -i /path/to/private/key username@instance-ip
   ```

### Nebius GPU Cluster Setup

1. Navigate to the cluster configuration directory:
   ```bash
   cd infrastructure/nebius/cluster
   ```

2. Follow the same steps 2-4 as for the single GPU setup, configuring cluster-specific parameters:
   - Number of nodes
   - Master node specifications
   - Worker node specifications
   - Network topology

3. After deployment, access the cluster entry point as shown in the Terraform output.

## Extending the Infrastructure

The repository is structured to easily accommodate additional cloud providers and resource types.

### Adding a New Cloud Provider

To add support for a new cloud provider (e.g., AWS, GCP, Azure):

1. Create a new directory under `infrastructure/`:
   ```bash
   mkdir -p infrastructure/aws
   ```

2. Follow the established pattern with subdirectories for resource types:
   ```bash
   mkdir -p infrastructure/aws/single_gpu
   mkdir -p infrastructure/aws/cluster
   ```

3. Create the necessary Terraform files in each directory:
   - `main.tf` - Main resource definitions
   - `variables.tf` - Input variables
   - `outputs.tf` - Output values
   - `terraform.tfvars.example` - Example variable values
   - `provider.tf` - Provider configuration

4. Document provider-specific requirements in a README.md file in the provider's directory.

### Implementation Guidelines for New Providers

When implementing infrastructure for a new cloud provider:

1. Follow the same logical structure of resources (compute, network, storage, etc.)
2. Maintain consistent variable naming where possible
3. Document provider-specific quirks or limitations
4. Include example configurations and connection instructions
5. Test thoroughly before committing

## Security Considerations

Follow these security best practices:

1. **Never commit sensitive credentials to Git**
   - Use `.gitignore` to exclude credential files
   - Use environment variables where possible

2. **State File Security**
   - Consider using remote state backends with proper access controls
   - All state files are git-ignored by default

3. **SSH Key Management**
   - Only reference public keys in your configurations
   - Never commit private SSH keys

4. **Verify Git Ignores Are Working**
   ```bash
   git check-ignore -v infrastructure/*/terraform.tfstate
   git check-ignore -v infrastructure/*/terraform.tfvars
   ```

5. **Audit Before Committing**
   ```bash
   git status
   git diff --cached
   ```

## Troubleshooting

- **Authentication Issues**: Verify your credentials in `terraform.tfvars` or environment variables
- **Resource Limits**: Check your cloud provider's quotas for GPU resources
- **Network Problems**: Ensure your firewall/security group rules allow required connections
- **State File Errors**: If Terraform state is corrupted, consider using state recovery options

## Contributing

When contributing new provider implementations:
1. Create a new branch for your provider
2. Follow the structure outlined above
3. Include comprehensive documentation
4. Create a pull request with a detailed description of your implementation
