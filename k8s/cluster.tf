data "aws_availability_zones" "available" {
  state = "available"
}


module "vpc" {
  source  = "terraform-aws-modules/vpc/aws"
  version = "~> 6.0.0"

  name = "protoplast-${var.env}"
  cidr = "10.0.0.0/16"
  azs  = slice(data.aws_availability_zones.available.names, 0, 2)

  public_subnets  = ["10.0.0.0/24", "10.0.1.0/24"]
  map_public_ip_on_launch = true
  enable_dns_hostnames = true
}


module "eks" {
  source          = "terraform-aws-modules/eks/aws"
  version         = "~> 21.0"

  name    = "protoplast-${var.env}"
  kubernetes_version = "1.33"
  endpoint_public_access = true
  enable_cluster_creator_admin_permissions = true

  subnet_ids         = module.vpc.public_subnets
  vpc_id          = module.vpc.vpc_id
  enable_irsa = true

  addons = {
    coredns                = {}
    eks-pod-identity-agent = {
      before_compute = true
    }
    kube-proxy             = {}
    vpc-cni                = {
      before_compute = true
    }
  }

  eks_managed_node_groups = {
    manager = {
      ami_type       = "AL2023_x86_64_STANDARD"
      instance_types = ["t3a.medium"]
      min_size     = 1
      max_size     = 2
      desired_size = 1
      disk_size = 20
    }
    cpu_worker = {
      ami_type = "AL2023_x86_64_STANDARD"
      instance_types = [var.cpu_instance_type]
      min_size     = var.cpu_min_size
      max_size     = var.cpu_max_size
      desired_size = 1
      disk_size = 100
      labels = {
        "gpu" = "false"
      }
    }
    gpu_worker = {
      ami_type = "AL2023_x86_64_STANDARD"
      instance_types = [var.gpu_instance_type]
      min_size     = var.gpu_min_size
      max_size     = var.gpu_max_size
      desired_size = 0
      disk_size = 100
      labels = {
        "gpu" = "true"
      }
    }
  }


}

resource "kubernetes_namespace" "protoplast" {
  metadata {
    name = "protoplast-${var.env}"
  }
}