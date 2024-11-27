{{ fullname | escape | underline}}

.. automodule:: {{ fullname }}
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

   {% block modules %}
   {% if modules %}
   .. rubric:: Submodules
   .. autosummary::
      :toctree:
      :recursive:
   {% for item in modules %}
      {{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}
