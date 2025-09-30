provider "aws" {
  region = "us-west-2"
}

data "aws_eks_cluster_auth" "protoplast" {
  name = module.eks.cluster_name
}

provider "kubernetes" {
  host                   = module.eks.cluster_endpoint
  cluster_ca_certificate = base64decode(module.eks.cluster_certificate_authority_data)
  token                  = data.aws_eks_cluster_auth.protoplast.token
}

provider "helm" {
  kubernetes = {
    host = module.eks.cluster_endpoint
    cluster_ca_certificate = base64decode(module.eks.cluster_certificate_authority_data)
    token = data.aws_eks_cluster_auth.protoplast.token
  }
}