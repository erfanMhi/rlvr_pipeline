locals {
  fs_device_name = "single-gpu-filesystem"
}

data "nebius_vpc_v1_subnet" "default" {
  parent_id = var.project_id
  name      = "default-subnet-jvzrrihy"
}

# Boot disk
resource "nebius_compute_v1_disk" "single_gpu_boot" {
  parent_id        = var.project_id
  type             = "NETWORK_SSD"
  block_size_bytes = 4096
  name             = "single-gpu-disk"
  size_bytes       = 214748364800 # 200 GiB
  source_image_family = {
    image_family = "ubuntu22.04-cuda12"
  }
}

#
# resource "nebius_compute_v1_filesystem" "single_gpu_fs" {
#   parent_id        = var.project_id
#   name             = local.fs_device_name
#   type             = "NETWORK_SSD"
#   block_size_bytes = 4096
#   size_bytes       = 107374182400 # 100 GiB
# }

# Single GPU instance
resource "nebius_compute_v1_instance" "single_gpu" {
  name      = "single-gpu-instance"
  parent_id = var.project_id

  network_interfaces = [
    {
      name              = "eth0"
      subnet_id         = data.nebius_vpc_v1_subnet.default.id
      ip_address        = {}
      public_ip_address = {}
    }
  ]

  resources = {
    platform = "gpu-h100-sxm"      # Using H100 GPU
    preset   = "1gpu-16vcpu-200gb" # Single GPU preset
  }

  boot_disk = {
    attach_mode = "READ_WRITE"
    existing_disk = {
      id = nebius_compute_v1_disk.single_gpu_boot.id
    }
  }

  # filesystems = [
  #   {
  #     attach_mode = "READ_WRITE",
  #     mount_tag   = local.fs_device_name
  #     existing_filesystem = {
  #       id = nebius_compute_v1_filesystem.single_gpu_fs.id
  #     }
  #   }
  # ]

  cloud_init_user_data = templatefile("${path.module}/scripts/cloud-init.tftpl", {
    vm_username    = var.vm_username
    ssh_public_key = file(var.vm_ssh_public_key_path)
    fs_device_name = local.fs_device_name
  })
}
