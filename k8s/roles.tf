variable "service_account_name" {
  type = string
}

data "aws_iam_openid_connect_provider" "eks" {
  url = data.aws_eks_cluster.ray.identity[0].oidc[0].issuer
}

data "aws_iam_policy_document" "trust" {
  statement {
    effect = "Allow"

    principals {
      type        = "Federated"
      identifiers = [data.aws_iam_openid_connect_provider.eks.arn]
    }

    actions = ["sts:AssumeRoleWithWebIdentity"]

    condition {
      test     = "StringEquals"
      variable = "${replace(data.aws_iam_openid_connect_provider.eks.url, "https://", "")}:sub"
      values   = ["system:serviceaccount:ray:${var.service_account_name}"]
    }
  }
}


resource "aws_iam_role_policy" "s3_access" {
    name = "s3_access"
    role = aws_iam_role.ray_head_role
    policy = data.aws_iam_policy_document.ray_s3_access
}

resource "aws_iam_role" "ray_head_role" {
  name = "protoplast-ray-head-role-${var.env}"
  assume_role_policy = data.aws_iam_policy_document.trust
}

resource "aws_iam_role" "ray_worker_role" {
    name = "protoplast-ray-worker-role-${var.env}"
    assume_role_policy = data.aws_iam_policy_document.trust
}

resource "aws_iam_role_policy" "head_s3_access" {
    name = "s3_access"
    role = aws_iam_role.ray_head_role
    policy = data.aws_iam_policy_document.ray_s3_access
}


resource "aws_iam_role_policy" "worker_s3_access" {
    name = "s3_access"
    role = aws_iam_role.ray_worker_role
    policy = data.aws_iam_policy_document.ray_s3_access
}
