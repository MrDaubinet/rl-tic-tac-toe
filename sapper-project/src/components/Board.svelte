<script>
  // Component
  import Marker from "./Marker.svelte"
  // Data prop
  export let state = [0, 0, 0, 0, 0, 0, 0, 0, 0]
  // Function prop
  export let selectMarker
</script>

<div class="grid grid-cols-3 gap-0">
  {#each state as value, index}
    <div class="col-span-1">
      <Marker 
        {value}
        {index}
        {selectMarker}
      />
    </div>
  {/each}
</div>

