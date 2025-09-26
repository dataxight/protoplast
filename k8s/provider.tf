provider "aws" {
  region = "us-west-2"
}

provider "kubernetes" {
  host                   = aws_eks_cluster.protoplast.endpoint
  cluster_ca_certificate = base64decode(aws_eks_cluster.protoplast.certificate_authority[0].data)
  token                  = data.aws_eks_cluster_auth.protoplast.token
}