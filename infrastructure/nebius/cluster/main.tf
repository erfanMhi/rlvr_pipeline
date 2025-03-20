locals {
  fs_device_name = "training-vm-filesystem-1"
  # master_ip = try(nebius_compute_v1_instance.training_h100_master.network_interfaces[0].ip_address.address, "")
}

data "nebius_vpc_v1_subnet" "default" {
  parent_id = var.project_id
  name      = "default-subnet-jvzrrihy"
}

# Boot disk for master
resource "nebius_compute_v1_disk" "training_boot_master" {
  parent_id           = var.project_id
  type                = "NETWORK_SSD"
  block_size_bytes    = 4096
  name                = "training-vm-disk-master"
  size_bytes          = 214748364800 # 200 GiB
  source_image_family = {
    image_family = "ubuntu22.04-cuda12"
  }
}

# Boot disks for workers
resource "nebius_compute_v1_disk" "training_boot_workers" {
  count              = var.cluster_size - 1
  parent_id          = var.project_id
  type               = "NETWORK_SSD"
  block_size_bytes   = 4096
  name               = "training-vm-disk-worker-${count.index}"
  size_bytes         = 214748364800 # 200 GiB
  source_image_family = {
    image_family = "ubuntu22.04-cuda12"
  }
}

resource "nebius_compute_v1_filesystem" "training_1024gib" {
  parent_id        = var.project_id
  name             = "training-vm-filesystem-1"
  type             = "NETWORK_SSD"
  block_size_bytes = 4096
  size_bytes       = 107374182400  # 100 GiB (100 * 1024 * 1024 * 1024 bytes)
}

resource "nebius_compute_v1_gpu_cluster" "training_fabric_3" {
  parent_id         = var.project_id
  name              = "training-demo-cluster"
  infiniband_fabric = "fabric-3"
}

# Master instance
resource "nebius_compute_v1_instance" "training_h100_master" {
  name  = "training-h100-master"

  parent_id = var.project_id
  gpu_cluster = {
    id = nebius_compute_v1_gpu_cluster.training_fabric_3.id
  }
  network_interfaces = [
    {
      name              = "eth0"
      subnet_id         = data.nebius_vpc_v1_subnet.default.id
      ip_address        = {}
      public_ip_address = {}
    }
  ]
  resources = {
    platform = "gpu-h100-sxm"
    preset   = "8gpu-128vcpu-1600gb"
  }
  boot_disk = {
    attach_mode   = "READ_WRITE"
    existing_disk = {
      id = nebius_compute_v1_disk.training_boot_master.id
    }
  }
  filesystems = [
    {
      attach_mode         = "READ_WRITE",
      mount_tag           = local.fs_device_name
      existing_filesystem = {
        id = nebius_compute_v1_filesystem.training_1024gib.id
      }
    }
  ]
  cloud_init_user_data = templatefile("${path.module}/scripts/cloud-init.tftpl", {
    vm_username         = var.vm_username
    ssh_public_key      = file(var.vm_ssh_public_key_path)
    fs_device_name      = local.fs_device_name
    instance_index     = 0
    cluster_size       = var.cluster_size
    master_ip          = ""
  })
}

# Worker instances
resource "nebius_compute_v1_instance" "training_h100_workers" {
  count = var.cluster_size - 1
  name  = "training-h100-${count.index}"

  depends_on = [nebius_compute_v1_instance.training_h100_master]

  parent_id = var.project_id
  gpu_cluster = {
    id = nebius_compute_v1_gpu_cluster.training_fabric_3.id
  }
  network_interfaces = [
    {
      name              = "eth0"
      subnet_id         = data.nebius_vpc_v1_subnet.default.id
      ip_address        = {}
      public_ip_address = {}
    }
  ]
  resources = {
    platform = "gpu-h100-sxm"
    preset   = "8gpu-128vcpu-1600gb"
  }
  boot_disk = {
    attach_mode   = "READ_WRITE"
    existing_disk = {
      id = nebius_compute_v1_disk.training_boot_workers[count.index].id
    }
  }
  filesystems = [
    {
      attach_mode         = "READ_WRITE",
      mount_tag           = local.fs_device_name
      existing_filesystem = {
        id = nebius_compute_v1_filesystem.training_1024gib.id
      }
    }
  ]
  cloud_init_user_data = templatefile("${path.module}/scripts/cloud-init.tftpl", {
    vm_username         = var.vm_username
    ssh_public_key      = file(var.vm_ssh_public_key_path)
    fs_device_name      = local.fs_device_name
    instance_index     = count.index + 1
    cluster_size       = var.cluster_size
    master_ip          = split("/", nebius_compute_v1_instance.training_h100_master.status.network_interfaces[0].ip_address.address)[0]
  })
}
