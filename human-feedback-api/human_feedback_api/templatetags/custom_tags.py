from base64 import b64encode
from django import template

register = template.Library()


@register.inclusion_tag('_comparison.html')
def _comparison(comparison, experiment):
    return {'comparison': comparison, "experiment": experiment}
