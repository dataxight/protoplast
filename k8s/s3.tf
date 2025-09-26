resource "aws_s3_bucket" "protoplast_bucket" {
  bucket = "protoplast-${var.aws_account_id}-${var.aws_region}"
  tags = {
    Environment = var.env
  }
  force_destroy = false  # set true if you want to allow deleting non-empty bucket
}


data "aws_iam_policy_document" "ray_s3_access" {
  statement {
    effect = "Allow"

    actions = [
      "s3:GetObject",
      "s3:PutObject",
      "s3:ListBucket"
    ]

    resources = [
      "arn:aws:s3:::${aws_s3_bucket.protoplast_bucket.bucket}",
      "arn:aws:s3:::${aws_s3_bucket.protoplast_bucket.bucket}/*"
    ]
  }
}
